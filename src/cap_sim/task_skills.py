"""Versioned task experience for existing planners; never changes the run loop."""
import copy
import fcntl
import os
import re
import tempfile
import time
from pathlib import Path
from .io import digest, read, write


def validate_draft(draft):
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,79}', draft.get('task_id', '')):
        raise ValueError('Invalid task_id')
    for key in ('skill_markdown', 'revision_reason'):
        if not isinstance(draft.get(key), str) or not draft[key].strip():
            raise ValueError('Missing ' + key)
    controls = draft.get('supported_controls')
    if not isinstance(controls, list) or not controls or not all(isinstance(c,str) for c in controls):
        raise ValueError('Declare supported controls')
    evidence = draft.get('evidence', [])
    ids = set()
    for item in evidence:
        if not item.get('id') or item['id'] in ids or not item.get('scope'):
            raise ValueError('Unique scoped evidence required')
        if item.get('status') not in ('observed', 'failed', 'inferred') or not re.fullmatch(r'[0-9a-f]{64}',item.get('sha256','')):
            raise ValueError('Evidence status and sha256 required')
        ids.add(item['id'])
    strategies = draft.get('strategies', [])
    if not strategies:raise ValueError('Declare strategies and their evidence status')
    names = set()
    for strategy in strategies:
        name = strategy.get('id')
        if not name or name in names:raise ValueError('Unique strategy ids required')
        names.add(name)
        if strategy.get('status') not in ('candidate', 'observed', 'rejected'):
            raise ValueError('Strategy status must distinguish candidates from observations')
        refs = strategy.get('evidence_ids', [])
        if not set(refs) <= ids or (strategy['status'] != 'candidate' and not refs):
            raise ValueError('Observed/rejected strategy requires evidence')
        if not set(strategy.get('required_controls', [])) <= set(controls):
            raise ValueError('Strategy uses undeclared controls')
    if not isinstance(draft.get('limitations'), list) or not draft['limitations']:
        raise ValueError('State skill limitations')


def validate_artifact(artifact):
    if artifact.get('artifact_kind') != 'task_skill':
        raise ValueError('Expected a versioned task skill')
    body = {k:v for k,v in artifact.items() if k != 'revision_sha256'}
    if digest(body) != artifact.get('revision_sha256'):
        raise ValueError('Task skill hash mismatch')
    validate_draft({'task_id':artifact['task_id'], 'revision_reason':artifact['revision_reason'],
                    **artifact['experience']})
    return artifact


def load_skill(path):
    path=Path(path);artifact=validate_artifact(read(path))
    md=path.parent/'skill.md'
    if md.exists() and md.read_text() != artifact['experience']['skill_markdown']:
        raise ValueError('Task skill Markdown hash/content mismatch')
    return artifact


def publish_skill(draft, registry, parent=None):
    """Create a new version; exact parent required for revision. Existing versions untouched."""
    validate_draft(draft)
    registry=Path(registry);registry.mkdir(parents=True,exist_ok=True)
    with (registry/'.publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        task_dir=registry/draft['task_id'];latest=task_dir/'latest.json'
        previous=read(latest) if latest.exists() else None
        if parent != (previous['revision_sha256'] if previous else None):
            raise ValueError('Revision parent must match current latest skill')
        if previous:load_skill(task_dir/previous['artifact'])
        version=previous['version']+1 if previous else 1
        experience={k:copy.deepcopy(v) for k,v in draft.items() if k not in ('task_id','revision_reason')}
        artifact={'schema_version':1,'artifact_kind':'task_skill','task_id':draft['task_id'],
                  'version':version,'parent_sha256':parent,'revision_reason':draft['revision_reason'],
                  'created_unix_s':time.time(),'experience':experience}
        artifact['revision_sha256']=digest(artifact)
        task_dir.mkdir(parents=True,exist_ok=True);dest=task_dir/f'v{version:04d}'
        if dest.exists():raise FileExistsError(dest)
        with tempfile.TemporaryDirectory(prefix='.build-',dir=task_dir) as temp:
            build=Path(temp)/'version';build.mkdir()
            write(build/'skill.json',artifact);(build/'skill.md').write_text(experience['skill_markdown'])
            os.replace(build,dest)
        write(latest,{'version':version,'revision_sha256':artifact['revision_sha256'],
                      'artifact':f'v{version:04d}/skill.json'})
        return artifact


def planner_context(request):
    """Make task-level choices explicit without executing or trusting skill pseudocode."""
    skills=[validate_artifact(s) for s in request.get('skills',[]) if s.get('artifact_kind')=='task_skill']
    if not skills:return request
    enriched=copy.deepcopy(request)
    actions = request.get('interface', {}).get('action')
    controls = set(actions) if isinstance(actions, dict) else None
    if controls is not None and 'native_prefix' in request.get('interface', {}):controls.add('native_prefix')
    options=[{'task_id':s['task_id'],**copy.deepcopy(option),
              'missing_runtime_controls':None if controls is None else sorted(set(option.get('required_controls',[]))-controls)}
             for s in skills for option in s['experience']['strategies']]
    enriched['task_skill_context']={
        'frozen_versions':[{'task_id':s['task_id'],'version':s['version'],'sha256':s['revision_sha256']} for s in skills],
        'strategy_options':options,
        'required_decision_fields':['strategy_id','applicability','approach','stance','grasp','next_observation_gate','abort_or_recovery'],
        'rules':['Choose strategy from current geometry and measured state; do not blindly replay a reference prefix.',
                 'Distinguish observed scope from untested candidates; controller capability does not imply successful transfer.',
                 'Use only declared runtime interface actions. Skill pseudocode is guidance, never executable authority.',
                 'Explain approach route, base stance/heading, wrist orientation, contact region and fingers in analysis.strategy.',
                 'Emit only the next bounded stage when replanning; observe after changing stance before choosing grasp targets.',
                 'Freeze this skill version for this attempt. Record failure notes; a new revision may be selected before the next attempt.']}
    return enriched


class EvolvingSkillPlanner:
    """Select latest task skills at attempt start; retain exact versions during feedback.

    This adapter does not change task physics, controls, run loop or retry budget.
    Each effective request and selected revision is recorded separately because the
    core's original experience.json remains the unchanged seed experience.
    """
    def __init__(self, planner, task_directories, audit_directory):
        self.planner=planner
        self.roots=[Path(p).resolve() for p in task_directories]
        self.audit=Path(audit_directory)

    def _selected(self, request):
        config={'task_directories':[str(p) for p in self.roots]}
        conf_path=self.audit/'resolver.json'
        if conf_path.exists() and read(conf_path)!=config:
            raise ValueError('Skill registry configuration changed on resume')
        if not conf_path.exists():write(conf_path,config)
        number=request['attempt'];snapshot=self.audit/f'a{number:03d}_skills.json'
        if snapshot.exists():
            saved=read(snapshot)
            for skill in saved['skills']:validate_artifact(skill)
            return saved['skills']
        if request.get('feedback_index',0):
            raise ValueError('Missing initial skill snapshot; refusing mid-attempt version selection')
        selected=[]
        for root in self.roots:
            pointer=read(root/'latest.json');path=(root/pointer['artifact']).resolve()
            if not path.is_relative_to(root):raise ValueError('Skill pointer escapes task registry')
            skill=load_skill(path)
            if skill['revision_sha256']!=pointer['revision_sha256'] or skill['version']!=pointer['version']:
                raise ValueError('Skill pointer hash/version mismatch')
            if any(s['task_id']==skill['task_id'] for s in selected):raise ValueError('Duplicate evolving task skill')
            selected.append(skill)
        write(snapshot,{'attempt':number,'selected_unix_s':time.time(),'skills':selected,
                        'previous_attempt':request.get('history',[])[-1:]})
        return selected

    def plan(self, request):
        selected=self._selected(request);enriched=copy.deepcopy(request)
        ids={s['task_id'] for s in selected}
        enriched['skills']=[s for s in enriched.get('skills',[]) if s.get('task_id') not in ids]+selected
        enriched=planner_context(enriched)
        number=request['attempt'];feedback=request.get('feedback_index',0)
        path=self.audit/f'a{number:03d}_f{feedback:02d}_request.json'
        if path.exists() and read(path)!=enriched:
            raise ValueError('Effective planner request changed for an existing decision')
        write(path,enriched)
        decision=self.planner.plan(enriched)
        decision['skill_usage']=[{'task_id':s['task_id'],'version':s['version'],
                                 'revision_sha256':s['revision_sha256'],
                                 'revision_reason':s['revision_reason']} for s in selected]
        decision['effective_request_path']=str(path.resolve())
        write(self.audit/f'a{number:03d}_f{feedback:02d}_decision.json',decision)
        return decision

    def reflect(self, request):
        # Reflection uses the completed attempt's frozen version, not latest.
        saved=read(self.audit/f"a{request['attempt']:03d}_f00_request.json")
        enriched=copy.deepcopy(request);enriched['skills']=saved['skills']
        for skill in enriched['skills']:
            if skill.get('artifact_kind')=='task_skill':validate_artifact(skill)
        enriched=planner_context(enriched)
        write(self.audit/f"a{request['attempt']:03d}_reflection_request.json",enriched)
        if not hasattr(self.planner,'reflect'):raise ValueError('Required reflection planner unavailable')
        result=self.planner.reflect(enriched)
        write(self.audit/f"a{request['attempt']:03d}_reflection_decision.json",result)
        return result
