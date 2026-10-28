"""Small genuine-contact MuJoCo integration demo, NOT a G1 substitute.
Two position-actuated Cartesian fingers lift a free dynamic sphere by friction.
"""
import math
import copy
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from ..io import write, read, digest

class Backend:
    interface = {'backend':'mujoco_gripper', 'stages':['approach','contact','lift','hold'],
                 'action':{'stage':'string','position':'world xyz m','opening':'inner finger gap m, .025..14','duration_s':'.05..8'},
                 'lift_gate':'both fingers contacting for .12 s; target displacement <= .031 m'}

    def describe(self, task):
        scene=task['scene'];xy=scene.get('object_position',[0.,0.]);radius=scene.get('radius',.03);z=scene.get('table_height',.65)+radius
        return {'object_position':[*xy,z], 'wrist_position':[*xy,z+.1], 'opening':.14,
                'radius':radius,'table_height':scene.get('table_height',.65), 'source':'initial scene definition, before stepping'}

    def reset(self, task, out):
        import mujoco
        self.mj=mujoco;self.task=task;self.out=Path(out)
        scene=task['scene']
        if set(scene)-{'object_position','radius','table_height'}:raise ValueError('Unknown scene variation')
        self.center=np.array(scene.get('object_position',[0.,0.]))
        if self.center.shape!=(2,) or not np.isfinite(self.center).all():raise ValueError('object_position must be finite xy')
        self.radius=float(scene.get('radius',.03));self.top=float(scene.get('table_height',.65))
        if not .015<=self.radius<=.05 or not .3<=self.top<=1.:raise ValueError('Demo scene outside bounds')
        x,y=self.center;z=self.top+self.radius
        xml=f'''<mujoco><option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast"/>
        <visual><global offwidth="640" offheight="480"/></visual>
        <worldbody><light pos="0 -1 2"/><geom name="table" type="box" pos="0 0 {self.top-.025}" size=".35 .35 .025" rgba=".3 .65 .7 1" friction=".7 .01 .001"/>
        <body name="apple" pos="{x} {y} {z}"><freejoint/><geom name="apple" type="sphere" size="{self.radius}" mass=".08" condim="6" friction="1 .01 .001" rgba=".8 .05 .03 1"/></body>
        <body name="left"><joint name="lx" type="slide" axis="1 0 0" damping="10"/><joint name="ly" type="slide" axis="0 1 0" damping="10"/><joint name="lz" type="slide" axis="0 0 1" damping="10"/><geom mass=".1" name="left" type="box" size=".01 .035 .025" condim="6" friction="1 .01 .001" rgba=".15 .15 .15 1"/></body>
        <body name="right"><joint name="rx" type="slide" axis="1 0 0" damping="10"/><joint name="ry" type="slide" axis="0 1 0" damping="10"/><joint name="rz" type="slide" axis="0 0 1" damping="10"/><geom mass=".1" name="right" type="box" size=".01 .035 .025" condim="6" friction="1 .01 .001" rgba=".15 .15 .15 1"/></body></worldbody><actuator><position joint="lx" kp="2000"/><position joint="ly" kp="2000"/><position joint="lz" kp="2000"/><position joint="rx" kp="2000"/><position joint="ry" kp="2000"/><position joint="rz" kp="2000"/></actuator></mujoco>'''
        if task.get('resources',{}).get('asset_manifest'):
            resources=task['resources'];path=Path(resources['asset_manifest']);manifest=read(path)
            from ..assets import task_from_assets
            task_from_assets(path,resources['object_id'],resources['support_id'])
            root=ET.parse(path.parent/manifest['scene']).getroot();finger_root=ET.fromstring(xml)
            asset_digest=digest({'manifest':manifest,'xml':ET.tostring(root).decode()})
            frozen=self.out.parents[1]/'asset_fingerprint.json'
            if frozen.exists() and read(frozen)!=asset_digest:raise RuntimeError('Assets changed within run')
            write(frozen,asset_digest)
            world=root.find('worldbody')
            body=world.find(f"body[@name='{resources['object_id']}']");body.set('name','apple');body.set('pos',f'{x} {y} {z}')
            geom=body.find('geom');geom.set('name','apple');geom.set('size',str(self.radius));geom.set('condim','6')
            table=world.find(f"body[@name='{resources['support_id']}']");g=table.find('geom');g.set('name','table')
            pos=list(map(float,table.get('pos').split()));pos[2]=self.top-float(g.get('size').split()[2]);table.set('pos',' '.join(map(str,pos)))
            for name in ('left','right'):world.append(copy.deepcopy(finger_root.find(f"worldbody/body[@name='{name}']")))
            root.append(copy.deepcopy(finger_root.find('actuator')));root.append(copy.deepcopy(finger_root.find('visual')))
            root.find('option').set('integrator','implicitfast')
            xml=ET.tostring(root,encoding='unicode')
        (self.out/'scene.xml').write_text(xml)
        self.m=mujoco.MjModel.from_xml_string(xml);self.d=mujoco.MjData(self.m)
        self.pos=np.array([x,y,z+.1]);self.opening=.14;self.z0=z;self.contact_time=0.;self.accept_hold=0.;self.samples=[];self.states=[];self.commands=[];self.forces=[]
        self._set();self.d.qpos[7:13]=self.d.ctrl;mujoco.mj_forward(self.m,self.d)
        return self.observe()

    def _set(self):
        self.d.ctrl[:3]=self.pos+[-self.opening/2-.01,0,0]
        self.d.ctrl[3:]=self.pos+[self.opening/2+.01,0,0]

    def contacts(self):
        vals={'left':0.,'right':0.,'support':0.,'table_robot':0.}
        for i,c in enumerate(self.d.contact):
            names={self.m.geom(int(c.geom1)).name,self.m.geom(int(c.geom2)).name}
            f=np.zeros(6);self.mj.mj_contactForce(self.m,self.d,i,f);force=max(float(f[0]),0.)
            if 'apple' in names:
                for side in ('left','right'):
                    if side in names:vals[side]+=force
                if 'table' in names:vals['support']+=force
            elif 'table' in names:vals['table_robot']+=force
        return vals

    def observe(self):
        return {'object_position':self.d.qpos[:3].tolist(),'wrist_position':self.pos.tolist(),
                'opening':self.opening,'contact_s':self.contact_time,'forces':self.contacts()}

    def execute(self, action, remaining):
        if set(action)-{'stage','position','opening','duration_s','name'}:raise ValueError('Unknown action channel')
        stage=action['stage'];target=np.asarray(action.get('position',self.pos),float);gap=float(action.get('opening',self.opening));duration=float(action['duration_s'])
        if stage not in ('approach','contact','lift','hold'):raise ValueError('Invalid stage')
        if target.shape!=(3,) or not np.isfinite(target).all() or not np.isfinite([gap,duration]).all():raise ValueError('Invalid target')
        if not .025<=gap<=.14 or not .05<=duration<=8 or np.any(np.abs(target[:2])>.30) or not self.top-.03<=target[2]<=self.top+.5:raise ValueError('Action outside workspace')
        if stage in ('lift','hold') and (self.contact_time<.12 or np.linalg.norm(target-self.pos)>.031):
            return {'completed':False,'reusable':False,'failure':'contact_or_increment_gate','samples':[],'observation':self.observe()}
        start=self.pos.copy();g0=self.opening;n=round(duration/.005);count=min(n,int((remaining+1e-9)/.005));rows=[];failure=None
        lost=0.
        for i in range(count):
            a=(i+1)/n;a=a*a*(3-2*a);self.pos=start+(target-start)*a;self.opening=g0+(gap-g0)*a;self._set()
            self.mj.mj_step(self.m,self.d);self.mj.mj_forward(self.m,self.d);c=self.contacts()
            opposed=c['left']>.1 and c['right']>.1;self.contact_time=self.contact_time+.005 if opposed else 0.;lost=0 if opposed else lost+.005
            row={'time':float(self.d.time),'dt':.005,'lift_m':float(self.d.qpos[2]-self.z0),'support_force':c['support'],'hand_force':c['left']+c['right'],'stable':bool(np.isfinite(self.d.qpos).all() and self.d.qpos[2]>self.top-.1)}
            rows.append(row);self.samples.append(row);self.states.append(np.r_[self.d.time,self.d.qpos,self.d.qvel]);self.commands.append(self.d.qpos[7:13].copy());self.forces.append(c['table_robot'])
            criteria=self.task['acceptance']
            good=row['lift_m']>=criteria['lift_m'] and row['support_force']<.1 and row['hand_force']>.05 and row['stable']
            self.accept_hold=self.accept_hold+.005 if good else 0.
            if self.accept_hold+1e-9>=criteria['hold_s']:break
            if not row['stable'] or (stage in ('lift','hold') and lost>=.03):failure='contact_lost_or_unstable';break
        completed=len(rows)==n
        if not failure and not completed and self.accept_hold+1e-9<self.task['acceptance']['hold_s']:failure='execution_deadline'
        return {'completed':completed,'reusable':completed and (stage=='approach' or self.contact_time>=.12),'failure':failure,'samples':rows,'observation':self.observe()}

    def export(self,out):
        np.savez_compressed(Path(out)/'states.npz',samples=np.asarray(self.states),gripper=np.asarray(self.commands))

    def metrics(self):
        p=np.asarray(self.commands)
        a=np.diff(p,2,axis=0)/(.005**2) if len(p)>2 else np.zeros((1,4))
        return {'roughness':float(np.sqrt(np.mean(a*a))), 'root_omega_rms':0., 'peak_table_force':max(self.forces,default=0.)}
