"""Explicit diagnostic admission and immutable historical-code recovery."""
from pathlib import Path
from .contracts import WorkflowError, file_hash, digest


def navigation_admission(request, artifact, scope, quality, limits):
    from .navigation.control import validate_route_calibration
    from .navigation.archive import verify_evidence
    from .parking import fit_stop_response
    if (not isinstance(artifact,dict) or artifact.get('schema_version')!=2 or artifact.get('kind')!='zero_intercept_velocity_response'
        or not isinstance(artifact.get('scope'),dict) or not isinstance(artifact.get('model'),dict)
        or not isinstance(artifact.get('samples'),list) or not artifact['samples']
        or not isinstance(artifact.get('evidence_sha256'),dict) or not artifact['evidence_sha256']):
        raise WorkflowError('invalid_input','Malformed stopping calibration artifact')
    verify_evidence(artifact['evidence_sha256'])
    try:
        fitted=fit_stop_response([row['velocity_xy_m_s'] for row in artifact['samples']],
                                [row['displacement_xy_m'] for row in artifact['samples']])
    except (KeyError,TypeError,ValueError) as exc:
        raise WorkflowError('integrity_error','Invalid stopping samples') from exc
    if digest(fitted)!=digest(artifact['model']):raise WorkflowError('integrity_error','Stopping fit mismatch')
    mode=request.get('navigation_execution','calibrated')
    if mode not in ('calibrated','uncalibrated_diagnostic'):
        raise WorkflowError('invalid_input','Unknown navigation execution mode')
    diagnostic=mode=='uncalibrated_diagnostic'
    reason=request.get('diagnostic_reason')
    if diagnostic and (not isinstance(reason,str) or not reason.strip()):
        raise WorkflowError('invalid_input','Diagnostic execution needs an explicit reason')
    try:
        model=validate_route_calibration(artifact,scope,quality,limits)
        return dict(mode=mode,calibrated=True,diagnostic_only=diagnostic,model=model)
    except WorkflowError as exc:
        if not diagnostic or exc.code!='needs_calibration':raise
        return dict(mode=mode,calibrated=False,diagnostic_only=True,reason_code=exc.code,
                    detail=exc.detail,diagnostic_reason=reason,
                    restrictions='One frozen attempt; unchanged controller, geometry/force/stability gates; not calibrated or training-ready')


def recover_template_evidence(template,repo):
    """Recover changed code only, from a byte-exact frozen snapshot; never assets."""
    repo=Path(repo).resolve()
    snapshot=template.get('pipeline',{}).get('git_commit_and_dirty_patch',{}).get('source_snapshot')
    snapshot=Path(snapshot).resolve() if snapshot else None
    hashes={};changes=[]
    for name,expected in template['preflight']['evidence_sha256'].items():
        path=Path(name).resolve()
        if path.is_file() and file_hash(path)==expected:
            hashes[name]=expected;continue
        if not path.is_relative_to(repo) or path.suffix!='.py' or snapshot is None or not snapshot.is_dir():
            raise WorkflowError('preflight_failed','Template evidence changed '+name)
        matches=[p for p in snapshot.rglob('*.py') if (p.name==path.name or p.name.endswith('_'+path.name))
                 and file_hash(p)==expected]
        if not matches:raise WorkflowError('preflight_failed','No exact archived source for '+name)
        archived=sorted(matches)[0];hashes[str(archived)]=expected
        changes.append(dict(original=name,archived=str(archived),sha256=expected,
                            current_sha256=file_hash(path) if path.exists() else None))
    return hashes,changes
