"""Compile an explicit model wait into audited fixed-target hold commands."""
from copy import deepcopy
from dataclasses import replace
import json
from .event_wait import validate_wait_call, execute_wait, prepare_wait


def submit_wait(session, call, backend, maximum_s):
    options = validate_wait_call(call, session.seen)
    if not session.ledger:
        raise ValueError('Issue an explicit act before waiting: no previous commanded target exists')
    command = deepcopy(session.ledger[-1]['command'])
    if command['pelvis_height_m'] < .72:
        raise ValueError('Standing wait requires last commanded pelvis height >= .72m')
    def status():
        current = backend.control_status()
        current['terminal'] = session.terminal
        current['remaining_sim_seconds'] = max(0.,maximum_s-float(backend.d.time))
        # Read-only actual endpoint identities are needed to distinguish a new
        # hand/object pair; the scalar LIVE contact feedback intentionally omits them.
        current['contacts'] = backend.contacts()
        current['command_contacts'] = session.receipts[-1]['latest_feedback'].get('command_contacts', {})
        return current

    prepare_wait(command,options['max_wait_s'],status)
    cid = call['call_id']
    session.seen.add(cid)
    session.tail.append(deepcopy(call))
    start = session.step
    before_time = float(backend.d.time)
    count = 0

    def once(fixed):
        nonlocal count
        captured = []
        original = session.convert
        marker = {'source':'executor_compiled_wait', 'model_call_id':cid, 'index':count}
        def convert(rows, state):
            if rows != [marker]:
                raise ValueError('Unexpected compiled wait rows')
            return [deepcopy(fixed)]
        def execute(value):
            result = backend.step(value, maximum_s-float(backend.d.time))
            captured.append(result)
            return result
        generated = {'type':'function_call', 'name':'act',
            'call_id':f'__compiled_wait__{cid}__{count}',
            'arguments':json.dumps({'actions':[marker], 'execution_note':'Executor compiled fixed-target hold from explicit wait_until_stable'})}
        count += 1
        session.convert = convert
        try:
            session.submit(generated, execute)
        finally:
            session.convert = original
        if not captured:
            raise RuntimeError('Compiled wait returned no execution evidence')
        return replace(captured[0], terminal=session.terminal or captured[0].terminal)

    try:
        report = execute_wait(command, options['max_wait_s'], status, once)
    except Exception as error:
        if session.terminal or session.step!=start or abs(float(backend.d.time)-before_time)>1e-9:
            session.terminal=True
            session.tail.append({'type':'function_call_output','call_id':cid,'output':json.dumps({
                'error':'wait_execution_or_feedback_failed','detail':str(error),'terminal':True,
                'confirmed_executed_steps':session.step-start,
                'execution_scope':'Per-command receipts retain confirmed evidence; uncertain execution is not fabricated.'})})
        raise
    report.update(model_call_id=cid, source='executor_compiled_wait',
                  executed_step_interval=[start,session.step], requested_max_wait_s=options['max_wait_s'])
    session.tail.append({'type':'function_call_output','call_id':cid,'output':json.dumps(report)})
    return report
