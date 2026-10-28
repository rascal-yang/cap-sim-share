"""Explicit robot/task/numerical settings; no experiment-tuned fallback values."""
from .contracts import WorkflowError, positive, canonical

REQUIRED={
    'navigation':('speed_m_s','waypoint_radius_m','stop_radius_m'),
    'settle':('window_s','drift_m','speed_m_s','yaw_speed_rad_s','foot_force_N','minimum_s','continuous_s'),
    'approach':('joint_speed_rad_s','wrist_speed_m_s','wrist_speed_rad_s'),
    'gates':('contact_force_N','contact_entry_s','contact_loss_s','collision_force_N','collision_duration_s','collision_instant_N',
             'fall_height_m','fall_tilt_rad','precontact_translation_m','precontact_rotation_rad'),
    'acceptance':('clearance_m','hold_duration_s','root_drift_m','support_force_N'),
    'budgets':('navigation_s','settle_s','approach_s','simulation_s')}


def validate_policy(p):
    canonical(p)
    try:
        positive(p['control_dt_s'],'control period')
        for group,keys in REQUIRED.items():
            for key in keys:positive(p[group][key],group+'.'+key)
        if p['settle']['minimum_s']>p['budgets']['settle_s'] or p['settle']['continuous_s']>p['budgets']['settle_s']:
            raise WorkflowError('invalid_input','Impossible settling budget')
        if p['gates']['collision_instant_N']<p['gates']['collision_force_N']:raise WorkflowError('invalid_input','Collision thresholds reversed')
        if p['compensation']['mode'] not in ('none','identified_equilibrium','bounded_integral'):
            raise WorkflowError('invalid_input','Unsupported compensation; no empirical fallback')
        if p['compensation']['mode']=='identified_equilibrium' and (not p['compensation'].get('calibrations') or not p['compensation'].get('scope')):
            raise WorkflowError('needs_calibration')
        if p['compensation']['mode']=='bounded_integral':
            for key in ('gain_per_s','cap_m'):positive(p['compensation'][key],key)
            if not p['compensation'].get('evidence_file'):raise WorkflowError('invalid_input','Feedback requires bound evidence')
        nav=p['navigation']
        if nav.get('standing_orientation') not in (None,'source','upright'):raise WorkflowError('invalid_input','Standing orientation')
        if nav.get('profile') not in (None,'distance_velocity','replay_navigation'):raise WorkflowError('invalid_input','Navigation profile')
        if nav.get('profile') in ('distance_velocity','replay_navigation'):
            for key in ('acceleration_m_s2','response_s'):positive(nav[key],key)
        if nav.get('profile')=='replay_navigation':
            from .navigation.control import RouteController
            from .navigation.prior import finite
            finite(nav['fingers_rad'],(14,),'Explicit navigation finger command')
            if not nav.get('control_evidence_file'):
                raise WorkflowError('needs_calibration','Scoped navigation control evidence required')
            RouteController([[0,0,0],[1,0,0]],nav['route_limits'])
            if type(nav.get('navigation_only')) is not bool or not nav.get('plan_file'):
                raise WorkflowError('invalid_input','Replay navigation requires bound plan and explicit navigation_only')
            if 'parking' in nav or 'correction' in nav or p.get('grasp_trigger'):
                raise WorkflowError('unsupported_capability','Replay route uses its own bounded controller; no legacy parking or GPT')
            if nav['route_limits']['timeout_s']>p['budgets']['navigation_s']:
                raise WorkflowError('invalid_input','Route timeout exceeds navigation budget')
        if 'whole_body_operation' in p:
            from .navigation.operation import validate_operation
            validate_operation(p['whole_body_operation'])
            if nav.get('profile')!='replay_navigation' or nav.get('navigation_only') is not False:
                raise WorkflowError('invalid_input','Whole-body operation requires explicit replay navigation execution')
            if p['compensation']['mode']!='none' or p.get('recovery',{}).get('max_retries',0):
                raise WorkflowError('unsupported_capability','Whole-body operation uses declared reference without legacy feedback/retries')
        elif nav.get('profile')=='replay_navigation' and nav.get('navigation_only') is False:
            raise WorkflowError('invalid_input','Missing whole_body_operation policy')
        if 'parking' in nav:
            park=nav['parking']
            if park['mode']!='predicted_stop' or nav.get('profile')!='distance_velocity':raise WorkflowError('invalid_input','Predictive parking requires smooth navigation')
            positive(park['trigger_radius_m'],'parking trigger radius')
            positive(park['max_residual_m'],'stopping model residual')
            if type(park['minimum_samples']) is not int or park['minimum_samples']<2:raise WorkflowError('invalid_input','At least two distinct stopping samples required')
            if not park.get('evidence_file'):raise WorkflowError('needs_calibration','Parking evidence required')
        if 'correction' in nav:
            corr=nav['correction']
            if type(corr.get('residual_feedback',False)) is not bool:raise WorkflowError('invalid_input','Residual feedback flag')
            if corr.get('method','walk') not in ('walk','planted_feet'):raise WorkflowError('invalid_input','Correction interface')
            if nav.get('profile')!='distance_velocity':raise WorkflowError('invalid_input','Correction requires smooth navigation')
            if type(corr['max_corrections']) is not int or corr['max_corrections']<0:raise WorkflowError('invalid_input','Correction count')
            for key in ('tolerance_m','minimum_progress_m','max_duration_s'):positive(corr[key],key)
        if 'grasp_trigger' in p:
            trigger=p['grasp_trigger']
            if trigger['provider']!='codex_exec' or not trigger['model'].startswith('gpt-6'):raise WorkflowError('invalid_input','Explicit GPT-6 grasp provider required')
            if type(trigger['max_calls']) is not int or trigger['max_calls']<1:raise WorkflowError('invalid_input','Grasp decision budget')
            for key in ('interval_s','approach_lookahead_s','timeout_s'):positive(trigger[key],key)
            if not trigger.get('executable') or not trigger.get('preflight_evidence_file'):raise WorkflowError('invalid_input','Grasp provider evidence required')
        recovery=p.get('recovery',{'max_retries':0})
        if type(recovery['max_retries']) is not int or recovery['max_retries']<0:raise WorkflowError('invalid_input','Recovery budget')
        if recovery.get('posture_restore',False):
            for key in ('joint_tolerance_rad','position_tolerance_m','angle_tolerance_rad'):positive(recovery[key],key)
        if recovery['max_retries']:
            for key in ('release_min_s','release_timeout_s','retreat_timeout_s'):positive(recovery[key],key)
            if recovery['release_min_s']>recovery['release_timeout_s']:raise WorkflowError('invalid_input','Recovery timing')
        planning=p['planning'];positive(planning['penetration_tolerance_m'],'penetration tolerance',zero=True)
        for group in ('ik','rrt','base_path'):
            if not isinstance(planning[group],dict):raise WorkflowError('invalid_input','Planning options')
            for k,v in planning[group].items():positive(v,group+'.'+k,zero=k=='seed')
        for group in (*REQUIRED,'planning','control_dt_s','compensation'):
            origin=p['origins'][group]
            if not isinstance(origin,dict) or set(origin)!={'kind','evidence'} or not isinstance(origin['evidence'],str) or not origin['evidence']:
                raise WorkflowError('invalid_input','Settings need provenance: '+group)
            if origin['kind'] not in ('robot_limit','task_requirement','numerical_setting','measured_calibration','explicit_disabled'):
                raise WorkflowError('invalid_input','Invalid settings provenance')
    except (KeyError,TypeError) as exc:
        raise WorkflowError('invalid_input','Incomplete execution profile: '+str(exc)) from None
    return p
