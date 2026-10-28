"""Narrow model tools operating on the caller's already supplied job."""
from pathlib import Path
from .contracts import require, validate_job
from .worker import start_job, reconcile_job
from ..workflow.contracts import file_hash, WorkflowError

SKILL = Path(__file__).parent/'skills/augmentation/SKILL.md'
PARAMETERS = {
    'read_skill': {'name': {'type': 'string', 'enum': ['augmentation']}},
    'validate_job': {}, 'submit_job': {},
    'get_job': {'job_id': {'type': 'string'}},
    'get_case': {'job_id': {'type': 'string'}, 'case_id': {'type': 'string'}},
    'export_job': {'job_id': {'type': 'string'}},
}
DESCRIPTIONS = {
    'read_skill': 'Read the augmentation workflow skill once, on demand.',
    'validate_job': 'Validate the supplied job and evidence without advancing physics.',
    'submit_job': 'Submit the complete supplied batch once; local Python executes it.',
    'get_job': 'Read compact job progress or its final summary.',
    'get_case': 'Read one case summary to explain a failure.',
    'export_job': 'Export recorded results without rerunning physics.',
}


def tool_definitions():
    return [{'type': 'function', 'name': name, 'description': DESCRIPTIONS[name], 'strict': True,
             'parameters': {'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False}}
            for name, props in PARAMETERS.items()]


def summary(row):
    return {key: row[key] for key in ('job_id', 'status', 'details', 'created_at', 'updated_at')} | {
        'cases': {name: {'stage': c['stage'], 'success': c['details'].get('success'),
                          'reason_code': c['details'].get('reason_code'),
                          'experiment_path': c['details'].get('experiment_path')}
                  for name, c in row['cases'].items()}}


def bound_inputs(job):
    proof = validate_job(job)
    hashes = dict(proof['input_hashes'])
    from ..workflow.service import implementation_files
    hashes.update({str(path): file_hash(path) for path in implementation_files().values()})
    hashes[str(SKILL.resolve())] = file_hash(SKILL)
    return hashes


def dispatch(name, arguments, context):
    require(name in PARAMETERS, 'Unknown tool', 'unsupported_tool')
    require(isinstance(arguments, dict) and set(arguments) == set(PARAMETERS[name]), 'Tool argument fields')
    for key in arguments:
        require(isinstance(arguments[key], str) and arguments[key], 'String tool argument required')
    if name == 'read_skill':
        require(arguments['name'] == 'augmentation', 'Unknown skill')
        context['skill_read'] = True
        return {'name': 'augmentation', 'content': SKILL.read_text(), 'sha256': file_hash(SKILL)}
    if name == 'validate_job':
        result = validate_job(context['job'])
        return {k: v for k, v in result.items() if k != 'input_hashes'} | {'bound_files': len(result['input_hashes'])}
    store = context['store']
    if name == 'submit_job':
        require(context.get('skill_read'), 'Read augmentation skill before submitting')
        hashes = bound_inputs(context['job'])
        submitted = store.submit(context['job'], hashes)
        context['job_id'] = submitted['job_id']
        started = start_job(store, submitted['job_id'])
        return submitted | {'worker': started}
    row = store.get(arguments['job_id'])
    require(row['job']['request_id'] == context['job']['request_id'], 'Job outside current request')
    if name == 'get_job': return summary(reconcile_job(store, row['job_id']))
    if name == 'get_case':
        require(arguments['case_id'] in row['cases'], 'Unknown case', 'not_found')
        return row['cases'][arguments['case_id']]
    try:
        from .export import export_job
    except ImportError:
        raise WorkflowError('unsupported_capability', 'Exporter unavailable') from None
    return export_job(store, row['job_id'])
