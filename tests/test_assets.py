import pytest
from cap_sim.assets import build_scene

def scene():
 return {'units':'m','up_axis':'z','objects':[{'id':'table','shape':'box','size':[1,.6,.05],'position':[0,0,.625],'dynamic':False,'rgba':[.3,.6,.7,1],'material':{'kind':'wood','roughness':.6},'texture':{'status':'unverified','source':None},'provenance':{'dimensions':'estimated','pose':'estimated','physics':'designed'}}]}

def test_unknown_scale_stays_uncalibrated(tmp_path):
 result=build_scene(scene(),tmp_path/'assets')
 assert result['calibrated'] is False
 assert (tmp_path/'assets/scene.xml').exists()

def test_invalid_dimensions_rejected(tmp_path):
 s=scene();s['objects'][0]['size'][0]=-1
 with pytest.raises(ValueError):build_scene(s,tmp_path/'assets')

def test_no_path_traversal_ids(tmp_path):
 s=scene();s['objects'][0]['id']='../bad'
 with pytest.raises(ValueError):build_scene(s,tmp_path/'assets')

@pytest.mark.parametrize('field,value',[('dynamic','false'),('shape','sphere')])
def test_ambiguous_geometry_rejected(tmp_path,field,value):
 s=scene();s['objects'][0][field]=value
 with pytest.raises(ValueError):build_scene(s,tmp_path/'assets')


def test_asset_task_executes_built_geometry(tmp_path):
 pytest.importorskip('mujoco')
 from cap_sim.assets import task_from_assets
 from cap_sim.backends.demo import Backend
 from cap_sim.planners import DemoPlanner
 from cap_sim.core import run_task
 s=scene();s['objects'].append({**s['objects'][0],'id':'fruit','shape':'sphere','size':[.06]*3,'position':[.02,.01,.68],'dynamic':True,'mass_kg':.08,'friction':[1,.01,.001]})
 build_scene(s,tmp_path/'assets')
 task=task_from_assets(tmp_path/'assets/manifest.json','fruit','table');task['budget']['attempts']=1
 result=run_task(task,DemoPlanner(),Backend(),tmp_path/'run')
 assert result['success']
 assert 'video_scene' in (tmp_path/'run/attempts/001/scene.xml').read_text()


def test_demo_stops_when_hold_criterion_is_reached(tmp_path):
 pytest.importorskip('mujoco')
 from cap_sim.backends.demo import Backend
 from cap_sim.planners import DemoPlanner
 from cap_sim.core import run_task
 from cap_sim.io import read
 task={'name':'stop','backend':'mujoco_gripper','scene':{},'acceptance':{'lift_m':.1,'hold_s':1.},'budget':{'attempts':1,'sim_seconds':15}}
 assert run_task(task,DemoPlanner(),Backend(),tmp_path/'run')['success']
 result=read(tmp_path/'run/attempts/001/result.json')
 assert result['failure'] is None
 assert result['acceptance']['max_hold_s']<=1.005
