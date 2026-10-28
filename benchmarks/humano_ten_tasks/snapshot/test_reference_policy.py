"""Synthetic regression check; uses only this experiment's own reset geometry."""
import json,tempfile,shutil
from pathlib import Path
import reference_policy as p
root=Path(__file__).resolve().parent;name='G1WholebodyTableApplePickTeleop'
q=json.loads((root/'mailbox/no_reference'/name/'a001_f00_request.json').read_text());o=q['observation']
s={'time':0.,'pose':o['root_pose']+o['body_joints'],'fingers':o['fingers'],'object':o['objects']['target']}
s2=json.loads(json.dumps(s));s2['time']=1.;s2['object'][2]+=.1
ref={'source_sha256':'synthetic-own-state','native_success_from_saved_states':False,'samples':[s,s2],'first_hand_contact_s':0.}
q.update(references=[ref],feedback_index=1,executed_actions=[],remaining_sim_seconds=29.);q['task']['scene']={'translation':[0,.1,0]}
with tempfile.TemporaryDirectory() as d:
 p.r=Path(d);dest=p.r/'runs/reference_augmented'/name/'attempts/001';dest.mkdir(parents=True);shutil.copy(root/'runs/no_reference'/name/'attempts/001/scene.mjb',dest/'scene.mjb')
 req=p.r/'a001_f01_request.json';req.write_text(json.dumps(q));p.respond(req);ans=json.loads(req.with_name('a001_f01_response.json').read_text());a=ans['actions'][0]
 assert a['stage']=='contact' and a['name']=='reference_to_0.000000'
 assert len(a['body_keyframes'][0])==37 and a['body_keyframes'][0][0]==0
 assert ans['planner_provenance']['live_model_call'] is False
 print('Synthetic no-contact gate and generated action schema passed.')
# A support contact on the opposite hand must not satisfy a bilateral object gate.
q=json.loads((root/'mailbox/no_reference/G1WholebodyGroundBoxPickTeleop/a001_f00_request.json').read_text());o=q['observation'];name=q['task']['name'];obj=o['objects']['target'];target_body=next(iter(o['object_details']['target']['bodies']))
s={'time':0.,'pose':o['root_pose']+o['body_joints'],'fingers':o['fingers'],'object':obj};s2=json.loads(json.dumps(s));s2['time']=1.;s2['object'][2]+=.1
q.update(references=[dict(ref,samples=[s,s2])],feedback_index=1,executed_actions=[],remaining_sim_seconds=29.);q['task']['scene']={'translation':[0,.1,0]};o['contact_s']=.2;o['contacts']['hand_object_N']=2.
o['contacts']['events']=[{'bodies':['left_hand_middle_0_link',target_body],'force_N':2.},{'bodies':['right_hand_middle_0_link','table'],'force_N':2.}]
with tempfile.TemporaryDirectory() as d:
 p.r=Path(d);dest=p.r/'runs/reference_augmented'/name/'attempts/001';dest.mkdir(parents=True);shutil.copy(root/'runs/no_reference'/name/'attempts/001/scene.mjb',dest/'scene.mjb')
 req=p.r/'a001_f01_request.json';req.write_text(json.dumps(q));p.respond(req);ans=json.loads(req.with_name('a001_f01_response.json').read_text());assert ans['actions'][0]['stage']=='contact';assert ans['analysis']['evidence'][-1]['contact_gate'] is False
 print('Opposite-hand table contact cannot open bilateral lift gate: passed.')
for sample in q['references'][0]['samples']:
 for side in ['left','right']:
  for suffix in ['_position','_quaternion_wxyz']:sample[side+suffix]=o[side+suffix]
q['attempt']=2;q['history']=[{'failure':'task_not_completed','observation':{'root_pose':o['root_pose']}}]
with tempfile.TemporaryDirectory() as d:
 p.r=Path(d);dest=p.r/'runs/reference_augmented'/name/'attempts/002';dest.mkdir(parents=True);shutil.copy(root/'runs/no_reference'/name/'attempts/001/scene.mjb',dest/'scene.mjb')
 req=p.r/'a002_f01_request.json';req.write_text(json.dumps(q));p.respond(req);ans=json.loads(req.with_name('a002_f01_response.json').read_text());a=ans['actions'][0]
 assert 'body_keyframes' not in a and sum(v*v for v in a['velocity_xy'])<=.4**2+1e-10
 assert a['stage']=='contact' and a['name']=='reference_to_0.000000'
 print('Failure-driven existing-interface switch preserves contact gate and velocity bounds: passed.')
q['executed_actions']=[{'name':'reference_to_1.000000'}];q['feedback_index']=3
with tempfile.TemporaryDirectory() as d:
 p.r=Path(d);req=p.r/'a002_f03_request.json';req.write_text(json.dumps(q));p.respond(req);a=json.loads(req.with_name('a002_f03_response.json').read_text())['actions'][0];assert a['stage']=='hold'
 q['observation']['contacts']['events'][1]['bodies'][1]=target_body
 req=p.r/'a002_f04_request.json';req.write_text(json.dumps(q));p.respond(req);a=json.loads(req.with_name('a002_f04_response.json').read_text())['actions'][0];assert a['stage']=='lift' and a['replan_after'];assert abs(a['right_position'][2]-o['right_position'][2]-.035)<1e-8
 print('Post-reference correction requires target contacts and advances only 3.5cm: passed.')
