"""Executed SONIC body tokens plus explicitly separate hand/actuator commands."""
from pathlib import Path
import numpy as np
from .contracts import WorkflowError, canonical, file_hash


def check(condition, detail):
    if not condition:
        raise WorkflowError('incomplete_sonic_record', detail)


def action_arrays(controller, commands, physics_dt, control_dt):
    check(physics_dt > 0 and control_dt >= physics_dt, 'Invalid recording periods')
    check(bool(controller) and bool(commands), 'No executed token/action samples')
    try:
        times=np.asarray([c['time_s'] for c in commands],float)
        ct=np.asarray([c['time'] for c in controller],float)
        check(np.isfinite(times).all() and np.isfinite(ct).all(), 'Nonfinite action times')
        check(np.allclose(np.diff(times),physics_dt,atol=1e-9,rtol=0), 'Missing physics action samples')
        check(np.all(np.diff(ct)>0) and abs(ct[0]-times[0])<1e-9, 'Controller start/order mismatch')
        # Forced policy ticks may occur between the regular50Hz ticks.
        indices=np.searchsorted(ct,times+1e-9,side='right')-1
        check(np.all(indices>=0) and np.all(times-ct[indices] < control_dt-1e-9), 'Physics steps without a current token')
        used=np.unique(indices)
        check(np.array_equal(used,np.arange(len(used))), 'Uncovered controller record')
        active=ct[used]
        grid=(active-times[0])/physics_dt
        check(np.allclose(grid,np.round(grid),atol=1e-7,rtol=0), 'Policy tick is not a physics tick')
        tokens=np.asarray([controller[i]['token'] for i in used],float)
        modes=np.asarray([controller[i]['mode'] for i in used])
        targets=np.asarray([controller[i]['target'] for i in used],float)
        body=np.asarray([c['body_target_rad'] for c in commands],float)
        hands=np.asarray([c['fingers_rad'] for c in commands],float)
        torques=np.asarray([c['ctrl_Nm'] for c in commands],float)
        check(tokens.ndim==2 and tokens.shape[1]>0 and np.isfinite(tokens).all(), 'Missing/nonfinite body token')
        check(modes.shape==(len(used),) and np.all(modes==1), 'Unexpected SONIC mode')
        check(targets.shape==(len(used),29) and body.shape==(len(commands),29), 'Body target dimensions')
        check(np.array_equal(body,targets[indices]), 'Executed body target differs from token decoder target')
        check(hands.shape==(len(commands),14) and torques.ndim==2 and len(torques)==len(commands) and torques.shape[1]>0,
              'Missing hand/actuator commands')
        check(all(np.isfinite(x).all() for x in (targets,body,hands,torques)), 'Nonfinite executed commands')
        return dict(time_s=times, token_time_s=active, tokens=tokens, mode=modes.astype(np.int64),
                    token_index=indices.astype(np.int64), body_target_rad=body, fingers_rad=hands, ctrl_Nm=torques,
                    stage=np.asarray([c['stage'] for c in commands],str),
                    source_row=np.asarray([c.get('source_index',c.get('source_row',-1)) for c in commands],np.int64),
                    unused_controller_indices=np.arange(len(used),len(controller),dtype=np.int64))
    except (ValueError,KeyError,TypeError,IndexError) as exc:
        raise WorkflowError('incomplete_sonic_record','Malformed token/action record: '+str(exc)) from exc


def save_action_bundle(output, controller, commands, physics_dt, control_dt, *, expected_physics_steps):
    check(type(expected_physics_steps) is int and len(commands)==expected_physics_steps, 'Missing executed physics step count coverage')
    arrays=action_arrays(controller,commands,physics_dt,control_dt)
    output=Path(output);path=output/'sonic_actions.npz';manifest=output/'sonic_actions.json'
    if path.exists() or manifest.exists():raise FileExistsError(path)
    with path.open('xb') as handle:np.savez_compressed(handle,**arrays)
    result=dict(schema_version=1,kind='executed_sonic_action_record',arrays_path=str(path.resolve()),
                arrays_sha256=file_hash(path),physics_steps=len(commands),token_count=len(arrays['tokens']),
                expected_physics_steps=expected_physics_steps,physics_dt_s=physics_dt,nominal_control_dt_s=control_dt,coverage_complete=True,
                fingers_encoded_in_token=False,body_target_post_decoder_modified=False,
                stages=sorted(set(arrays['stage'].tolist())),
                unused_controller_indices=arrays['unused_controller_indices'].tolist(),
                reproduction='Tokens require measured proprioceptive history and original decoder model; hands are separate. State/controller/model files remain required.')
    with manifest.open('xb') as handle:handle.write(canonical(result))
    return result
