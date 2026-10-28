"""Frozen numeric API execution of precision navigation and checkpoint branches."""
import json
from pathlib import Path
import sys
import time
from .agent import save
from .contracts import require
from ..workflow.contracts import file_hash, now


def navigation_profile(gain, resume=None):
    nav = dict(route_limits=dict(speed_m_s=.5, stop_radius_m=.025), transition_s=4.,
               simulation_budget_s=60., feedback_gain=gain, orientation_feedback_gain=1.,
               parking=dict(acceleration_m_s2=.5, response_s=.6, arrival_radius_m=.1,
                            settle_s=3., navigation_s=16., max_corrections=2, tolerance_m=.03,
                            max_translation_m=.03, support_frame='source_world', leg_mode='source_replay'),
               retry=dict(max_retries=0, release_s=.3, restore_s=3., joint_speed_rad_s=.5,
                          settle_s=1., joint_tolerance_rad=.13), checkpoint=dict(enabled=True))
    if resume:
        nav['checkpoint']['resume'] = resume
    return nav


def summarize_contacts(observations):
    pairs = {}
    for observation in observations:
        seen = set()
        for event in observation['contacts']['events']:
            if 'table' not in event['bodies']:
                continue
            pair = ' / '.join(event['bodies'])
            row = pairs.setdefault(pair, dict(samples=0, peak_N=0., first_time_s=observation['time_s']))
            if pair not in seen:
                row['samples'] += 1
                seen.add(pair)
            row['peak_N'] = max(row['peak_N'], event['force_N'])
    for row in pairs.values():
        row['duration_s'] = row['samples'] * .005
    return pairs


def clean_success(result):
    return bool(result['completed'] and result['failure'] is None and result['native_success']
                and result['retained_grasp'] and result['max_obstacle_N'] <= 5)


def cycle(request_path):
    from . import direct_grasp, replay_navigation, lateral_delivery
    request_path = Path(request_path)
    request = json.loads(request_path.read_text())
    started = time.monotonic()
    proof = request['provenance']
    proof['evidence_sha256'][str(request_path)] = file_hash(request_path)
    case, adapter, cfg = direct_grasp.prepare(Path(request['root']), Path(request['plan']),
        'apple_api_navigation', request['feedback_gain'], request['wrist_raise_m'], proof,
        keep_adapter=True, navigation=navigation_profile(request['feedback_gain'], request['resume']))
    result = replay_navigation.run(case, (adapter, cfg))
    cycle_s = time.monotonic() - started
    tick = time.monotonic()
    audit = lateral_delivery.audit(case, scene_only=True)
    require(audit['recorded_native_matches'] and audit['decoder_targets_unmodified'] and audit['contract_sha_valid'],
            'Independent physical evidence mismatch', 'integrity_error')
    pairs = summarize_contacts(json.loads((case/'execution/observations.json').read_text()))
    save(case/'contact_diagnosis.json', pairs)
    timing = dict(cycle_wall_s=cycle_s, audit_wall_s=time.monotonic()-tick, render_wall_s=0.,
                  rendered=False, perception_mode='privileged_only', ended_at=now())
    save(case/'delivery_timing.json', timing)
    manifest = json.loads((case/'execution/navigation_checkpoint_manifest.json').read_text())
    summary = dict(case=str(case), native_success=result['native_success'],
        clean_success=clean_success(result),
        retained_grasp=result['retained_grasp'], physics_steps=result['physics_steps'], completed=result['completed'],
        failure=result['failure'], max_obstacle_N=result['max_obstacle_N'], max_hand_object_N=result['max_hand_object_N'],
        table_contact_s=result['obstacle_contact_gt_0p1_s'], contact_diagnosis=pairs,
        navigation_executed=result['navigation_executed'], navigation_arrived=result['navigation_arrived'],
        available_checkpoints=list(manifest['checkpoints']), checkpoint_manifest_sha256=file_hash(case/'execution/navigation_checkpoint_manifest.json'),
        execution_audit=result['audit'], timing=timing,
        result_sha256=file_hash(case/'result.json'), audit_sha256=file_hash(case/'independent_audit.json'))
    save(request_path.with_suffix('.result.json'), summary)
    return summary


if __name__ == '__main__':
    cycle(sys.argv[1])
