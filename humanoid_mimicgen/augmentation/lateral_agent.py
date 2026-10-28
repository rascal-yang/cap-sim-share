"""Actual Responses tool loop; model decisions and physical results stay distinct."""
import base64
import json
from pathlib import Path
import time
import os

from .agent import attempt_stats, save
from .contracts import require
from .lateral_tools import ToolSession, SKILL, PARAMETERS, tool_definitions, validate_raise
from .provider import ResponsesProvider, validate_config
from ..workflow.contracts import WorkflowError, canonical, digest, file_hash, now

INSTRUCTIONS = '''You choose one next action for an apple grasp experiment in MuJoCo simulation.
Return exactly one registered tool call per response. The host executes it and sends
the result in the next request. Follow the supplied next-step hint; do not plan
across observations you have not received. Read the skill and use measured evidence.
Python plans references and runs SONIC mode1 physics; you choose bounded parameters
and reflect on results. No arbitrary shell. Native lift and collision quality are
separate. Accept only when both pass. Stop if the needed repair is unsupported.
After an accept/stop reflection, return a short honest summary. Initialized operating
stance does not validate navigation. Historical success does not complete a new test.'''


def next_step(state, complete=False):
    if complete:
        return 'Return the final measured summary; no further tool call.'
    if not state.get('skill_read'):
        return 'Call read_skill with {} now.'
    if not state.get('inspected'):
        return 'Call inspect_scene with {} now.'
    if not state.get('plan_file'):
        return 'Call plan_lateral with {} now.'
    rows = list(state.get('prepared', {}).items())
    if not rows or rows[-1][1].get('reflection', {}).get('action') == 'retry':
        return 'Call prepare_grasp: select the returned checked candidate_id, wrist_raise_m in [0,.03], and concise rationale.'
    ident, row = rows[-1]
    if not row.get('result'):
        return 'View the returned initial image, then call execute_grasp for prepared_id '+ident+' with your initial_review.'
    return 'Read the measured result and images, then call reflect_grasp for prepared_id '+ident+' with action/rationale/next_change.'


def normalize_json_action(raw):
    """Explicit protocol adapter; preserve the actual raw message in response.json."""
    require(isinstance(raw, dict) and raw.get('status') == 'completed', 'Incomplete JSON decision response', 'invalid_api_response')
    require(isinstance(raw.get('output'), list) and all(isinstance(x, dict) for x in raw['output']),
            'Malformed JSON response envelope', 'invalid_api_response')
    for item in raw['output']:
        require(item.get('type') in ('message', 'reasoning'), 'Unexpected native item in JSON protocol', 'invalid_api_response')
        content = item.get('content', [])
        require(isinstance(content, list) and all(isinstance(c, dict) for c in content), 'Malformed response content', 'invalid_api_response')
        require(not any(c.get('type') == 'refusal' for c in content), 'Model refused', 'api_refusal')
        if item.get('type') == 'message':
            require(all(c.get('type') == 'output_text' and isinstance(c.get('text'), str) for c in content),
                    'Unexpected JSON message content', 'invalid_api_response')
    texts = [c['text'] for item in raw.get('output', []) if item.get('type') == 'message'
             for c in item.get('content', []) if c.get('type') == 'output_text']
    try:
        action = json.loads('\n'.join(texts))
    except (ValueError, TypeError):
        raise WorkflowError('invalid_api_response', 'Expected one JSON action object') from None
    require(isinstance(action, dict) and set(action) == {'tool', 'arguments'} and isinstance(action['arguments'], dict),
            'JSON action fields', 'invalid_api_response')
    if action['tool'] == 'finish':
        require(set(action['arguments']) == {'summary'} and isinstance(action['arguments']['summary'], str),
                'JSON finish summary', 'invalid_api_response')
        output = []
    else:
        output = [dict(type='function_call', name=action['tool'], arguments=canonical(action['arguments']).decode(),
                       call_id='json_'+digest(raw))]
    return dict(status='completed', output=output, normalization='explicit_json_action_protocol',
                raw_model_response_sha256=digest(raw))


def json_protocol_items(items):
    converted = []
    for item in items:
        if item.get('type') == 'function_call':
            converted.append(dict(role='assistant', content=[dict(type='output_text', text=canonical(
                dict(tool=item['name'], arguments=json.loads(item['arguments']))).decode())]))
        elif item.get('type') == 'function_call_output':
            converted.append(dict(role='user', content=[dict(type='input_text', text=canonical(
                dict(tool_result_for=item['call_id'], result=json.loads(item['output']))).decode())]))
        elif 'role' in item:
            value = dict(item)
            if isinstance(value.get('content'), str):
                value['content'] = [dict(type='input_text', text=value['content'])]
            converted.append(value)
        # Reasoning items are not synthesized as native function-calling history.
    return converted


def apply_response(raw, session, root, provenance, items):
    require(isinstance(raw, dict) and raw.get('status') == 'completed', 'Incomplete model response', 'invalid_api_response')
    output = raw.get('output')
    require(isinstance(output, list) and all(isinstance(x, dict) for x in output), 'Missing output items', 'invalid_api_response')
    require(not any(c.get('type') == 'refusal' for x in output for c in x.get('content', [])), 'Model refused', 'api_refusal')
    actions = [x for x in output if x.get('type') == 'function_call']
    if not actions:
        require(session.complete(), 'Model ended before verified execution and accept/stop reflection', 'incomplete_workflow')
        items.extend(output)
        return True
    parsed = []
    seen = {}
    for action in actions:
        ident = action.get('call_id')
        require(isinstance(ident, str) and bool(ident), 'Missing tool call ID', 'invalid_api_response')
        try:
            args = json.loads(action['arguments'])
        except (KeyError, TypeError, ValueError):
            raise WorkflowError('invalid_api_response', 'Malformed tool JSON') from None
        name = action.get('name')
        require(name in getattr(session, 'allowed_parameters', PARAMETERS) and isinstance(args, dict)
                and set(args) == set(getattr(session, 'allowed_parameters', PARAMETERS)[name]),
                'Malformed tool name/fields', 'invalid_api_response')
        for key, value in args.items():
            if key == 'wrist_raise_m':
                validate_raise(value)
            elif key == 'feedback_gain':
                require(type(value) in (int, float) and 0 <= value <= 3, 'Malformed feedback gain', 'invalid_api_response')
            elif key == 'parent_prepared_id':
                require(isinstance(value, str), 'Malformed parent ID', 'invalid_api_response')
            else:
                require(isinstance(value, str) and bool(value.strip()), 'Malformed tool text', 'invalid_api_response')
        if name == 'reflect_grasp':
            require(args['action'] in ('accept', 'retry', 'stop'), 'Malformed reflection action', 'invalid_api_response')
        signature = digest(dict(name=name, arguments=args))
        require(ident not in seen or seen[ident] == signature, 'Tool ID reused with changed arguments', 'integrity_error')
        seen[ident] = signature
        cached = Path(root) / 'tool_calls' / (digest(ident) + '.json')
        if cached.exists():
            require(json.loads(cached.read_text())['signature'] == signature, 'Reused tool call ID changed arguments', 'integrity_error')
        parsed.append((action, args, signature))
    # Only observations in this request were visible when this response was authored.
    observed_prepared = []
    observed_results = []
    for item in items:
        if item.get('type') != 'function_call_output':
            continue
        try:
            prior_result = json.loads(item['output'])
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(prior_result, dict):
            continue
        ident = prior_result.get('prepared_id')
        if prior_result.get('initial_image') and ident in session.state['prepared']:
            observed_prepared.append(ident)
        observed_results.extend(ident for ident, row in session.state['prepared'].items()
                                if row.get('result') == prior_result)
    items.extend(output)
    calls = Path(root) / 'tool_calls'
    calls.mkdir(exist_ok=True)
    for action, args, signature in parsed:
        ident = action['call_id']
        path = calls / (digest(ident) + '.json')
        if path.exists():
            record = json.loads(path.read_text())
            require(record['signature'] == signature, 'Reused tool call ID changed arguments', 'integrity_error')
            result = record['result']
        else:
            started = time.monotonic()
            proof = dict(provenance, call_id=ident, tool=action.get('name'), arguments=args,
                         observed_prepared_ids=observed_prepared, observed_result_ids=observed_results)
            try:
                result = session.dispatch(action.get('name'), args, proof)
            except WorkflowError as exc:
                result = dict(status='error', reason_code=exc.code, detail=exc.detail)
            save(path, dict(signature=signature, name=action.get('name'), arguments=args, result=result,
                            provenance=proof, wall_s=time.monotonic()-started, ended_at=now()))
        items.append(dict(type='function_call_output', call_id=ident, output=canonical(result).decode()))
        images = ([result['initial_image']] if result.get('initial_image') else []) + result.get('review_images', [])
        if images:
            content = [dict(type='input_text', text='Bound tool image evidence for '+str(result.get('prepared_id', result.get('case'))))]
            for image_path in images:
                image_path = Path(image_path)
                mime = 'image/png' if image_path.suffix == '.png' else 'image/jpeg'
                content.append(dict(type='input_image', image_url='data:'+mime+';base64,'+base64.b64encode(image_path.read_bytes()).decode()))
            items.append(dict(role='user', content=content))
            if result.get('initial_image'):
                session.expose_image(result['prepared_id'])
    return False


def run_session(root, config, session, provider):
    root = Path(root).resolve()
    api = root / 'api'
    api.mkdir(exist_ok=True)
    require(not (root / 'agent_result.json').exists(), 'Completed session already exists; inspect saved evidence')
    started = now()
    clock = time.monotonic()
    live = isinstance(provider, ResponsesProvider) and getattr(provider, 'real_transport', True)
    protocol = config.get('decision_protocol', 'native_tools')
    require(protocol in ('native_tools', 'json_action'), 'Explicit decision protocol')
    history = config.get('history_protocol', 'native_items')
    require(history in ('native_items', 'role_messages'), 'Explicit history protocol')
    schemas = session.tool_definitions() if hasattr(session, 'tool_definitions') else tool_definitions()
    definitions = schemas if protocol == 'native_tools' else []
    instructions = getattr(session, 'instructions', INSTRUCTIONS)
    if protocol == 'json_action':
        instructions += ('\nThis session explicitly uses JSON actions, not native function calling. Return ONLY one JSON object '
            '{"tool":"registered_name","arguments":{...}}. After accept/stop reflection use '
            '{"tool":"finish","arguments":{"summary":"honest measured outcome"}}.\nAvailable tool schemas:\n'
            + canonical(schemas).decode())
    save(root / 'session.json', dict(config=config, provider=provider.config,
                                   transport='responses' if live else 'injected', started_at=started))
    visible = {k: config[k] for k in ('offset_m', 'max_attempts', 'decision_protocol') if k in config}
    visible.update(initialization='generated operation stance with stationary history; no navigation',
                   original_actions_visible=True)
    visible.update(getattr(session, 'model_context', {}))
    items = [dict(role='user', content=canonical(dict(task='apple lateral physical grasp', configuration=visible)).decode())]
    try:
        if config.get('bootstrap_read_skill_probe'):
            require(protocol == 'native_tools', 'Read-skill prefix requires native protocol')
            prefix = Path(config['bootstrap_read_skill_probe']).resolve()
            request = json.loads((prefix/'request.json').read_text())
            raw = json.loads((prefix/'response.json').read_text())
            timing = json.loads((prefix/'result.json').read_text())
            actions = [x for x in raw.get('output', []) if x.get('type') == 'function_call']
            require(raw.get('status') == 'completed' and timing.get('status') == 'completed'
                    and raw.get('model') == provider.config['model']
                    and request['provider']['model'] == provider.config['model']
                    and request['tools'] == tool_definitions() and len(actions) == 1
                    and actions[0].get('name') == 'read_skill' and json.loads(actions[0]['arguments']) == {},
                    'Only the actual completed read_skill call can be inherited', 'integrity_error')
            origin = {str(p): file_hash(p) for p in (prefix/'request.json', prefix/'response.json', prefix/'result.json')}
            save(api/'bootstrap_origin.json', dict(kind='inherited_actual_read_skill_call', source_sha256=origin))
            attempts = api/'attempts'
            attempts.mkdir(exist_ok=True)
            inherited = attempts/'000000.json'
            if not inherited.exists():
                save(inherited, dict(started_at=timing['started_at'], ended_at=timing['ended_at'],
                    wall_s=timing['wall_s'], status='responded', response_status='completed', usage=raw.get('usage'),
                    inherited_actual_request=True, source_sha256=origin))
            items = request['input'].copy()
            apply_response(raw, session, api, dict(transport='responses' if live else 'injected',
                decision_protocol='native_tools', native_function_calling=True, model=raw['model'],
                reported_model=raw['model'], api_calls_before_launch=attempt_stats(api)['api_calls'],
                evidence_sha256=origin, inherited_read_skill_only=True), items)
            items.append(dict(role='user', content='The transport check has ended. Continue with this authorized physical experiment: '
                              +canonical(dict(task='apple lateral physical grasp', configuration=visible)).decode()))
        for number in range(config['max_rounds']):
            require(time.monotonic()-clock < config['wall_s'], 'Session wall budget exhausted', 'agent_timeout')
            items.append(dict(role='user', content='Continue the authorized physical simulation one step at a time. '
                              +(session.next_step() if hasattr(session, 'next_step') else next_step(session.state, session.complete()))))
            folder = api / f'{number:04d}'
            folder.mkdir(exist_ok=True)
            sent_items = json_protocol_items(items) if protocol == 'json_action' or history == 'role_messages' else items
            request = dict(input=sent_items, tools=definitions, instructions=instructions,
                           decision_protocol=protocol, history_protocol=history)
            request_path = folder / 'request.json'
            if request_path.exists():
                require(json.loads(request_path.read_text()) == request, 'Saved API context differs', 'integrity_error')
            else:
                save(request_path, request)
            response_path = folder / 'response.json'
            if response_path.exists():
                raw = json.loads(response_path.read_text())
            else:
                attempts = api / 'attempts'
                attempts.mkdir(exist_ok=True)
                raw = None
                retry_limit = max(3, len(getattr(provider, 'slots', [])))
                origin = {}
                for retry in range(retry_limit):
                    remaining = config['wall_s']-(time.monotonic()-clock)
                    require(remaining > 0, 'Session deadline', 'agent_timeout')
                    if live:
                        provider.config['timeout_s'] = min(provider.config['timeout_s'], remaining)
                    attempt_path = attempts / f'{len(list(attempts.glob("*.json"))):06d}.json'
                    attempt = dict(started_at=now(), status='started', usage=None, wall_s=None)
                    save(attempt_path, attempt)
                    tick = time.monotonic()
                    try:
                        raw = provider.respond(sent_items, definitions, instructions)
                        attempt.update(status='responded', response_status=raw.get('status'), usage=raw.get('usage'),
                                       transport_runtime=getattr(provider, 'last_transport_runtime', None))
                    except WorkflowError as exc:
                        attempt.update(status='failed', reason_code=exc.code)
                        if exc.code != 'api_unavailable' or retry == retry_limit-1:
                            raise
                    finally:
                        origin = dict(credential_slot=getattr(provider, 'last_provider_label', None),
                                      credential_sequence=getattr(provider, 'last_provider_sequence', None),
                                      credential_endpoint=getattr(provider, 'last_provider_endpoint', provider.config.get('base_url')))
                        attempt.update(ended_at=now(), wall_s=time.monotonic()-tick,
                                       transport_runtime=getattr(provider, 'last_transport_runtime', None), **origin)
                        save(attempt_path, attempt)
                    if raw is not None:
                        break
                require(raw is not None, 'No API response', 'api_unavailable')
                save(folder/'response_origin.json', dict(origin, response_sha256=digest(raw)))
                save(response_path, raw)
            origin_path = folder/'response_origin.json'
            require(origin_path.exists() or not getattr(provider, 'slots', None),
                    'Cached API pool response has no bound credential origin', 'integrity_error')
            origin = json.loads(origin_path.read_text()) if origin_path.exists() else {}
            if origin:
                require(origin.get('response_sha256') == digest(raw), 'Response/origin binding changed', 'integrity_error')
            proof = dict(transport='responses' if live else 'injected', decision_protocol=protocol,
                         **origin,
                         history_protocol=history,
                         native_function_calling=protocol == 'native_tools', model=provider.config['model'],
                         reported_model=raw.get('model'), api_calls_before_launch=attempt_stats(api)['api_calls'],
                         evidence_sha256=dict(config.get('input_sha256', {}),
                             **{str(p): file_hash(p) for p in (request_path, response_path, SKILL,
                                  *([origin_path] if origin_path.exists() else []))}))
            dispatched = normalize_json_action(raw) if protocol == 'json_action' else raw
            if protocol == 'json_action':
                save(folder / 'normalized_action.json', dispatched)
            finished = apply_response(dispatched, session, api, proof, items)
            if finished or (getattr(session, 'finish_after_reflection', False) and session.complete()):
                result = dict(status='completed', started_at=started, ended_at=now(), wall_s=time.monotonic()-clock,
                              transport='responses' if live else 'injected', live_api_verified=live,
                              model=provider.config['model'], reasoning_effort=provider.config.get('reasoning_effort'),
                              decision_protocol=protocol, native_function_calling=protocol == 'native_tools',
                              history_protocol=history,
                              **attempt_stats(api), cases=[v['result'] | {'reflection': v['reflection']}
                                                         for v in session.state['prepared'].values()],
                              final_model_output=raw['output'], navigation_executed=any(v['result'].get('navigation_executed', False) for v in session.state['prepared'].values()),
                              operation_kind='trajectory_retargeting', original_actions_visible=True)
                save(root / 'agent_result.json', result)
                return result
        raise WorkflowError('agent_budget_exceeded', 'Model round limit exhausted')
    except Exception as exc:
        save(root / ('failure_'+str(len(list(root.glob('failure_*.json'))))+'.json'),
             dict(at=now(), reason_code=exc.code if isinstance(exc, WorkflowError) else type(exc).__name__,
                  **attempt_stats(api), elapsed_wall_s=time.monotonic()-clock))
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--provider', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    require(type(config['max_attempts']) is int and 1 <= config['max_attempts'] <= 3, 'Attempt budget 1..3')
    require(type(config['max_rounds']) is int and 1 <= config['max_rounds'] <= 20, 'Model round budget 1..20')
    require(type(config['offset_m']) in (int, float) and abs(config['offset_m']) <= .2, 'Lateral offset <=20cm')
    provider_config = validate_config(json.loads(Path(args.provider).read_text()))
    root = Path(config['output_root']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    profiles = os.environ.get('LATERAL_API_POOL_CONFIG')
    if profiles:
        from .api_pool import RotatingResponsesProvider
        values = json.loads(profiles)
        require(json.loads((root/'api_pool.json').read_text()) == values, 'Bound API pool differs')
        provider = RotatingResponsesProvider.from_profiles(provider_config, values, root/'api_routing.json')
    else:
        provider = ResponsesProvider(provider_config)
    from .suite_lock import exclusive
    with exclusive(root, 'lateral_agent'):
        if config.get('execution_mode') == 'privileged_fast':
            from .privileged_lateral import PrivilegedSession
            session = PrivilegedSession(root, config)
        else:
            session = ToolSession(root, config)
        print(canonical(run_session(root, config, session, provider)).decode())


if __name__ == '__main__':
    main()
