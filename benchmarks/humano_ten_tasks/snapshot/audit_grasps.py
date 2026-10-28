"""Supplementary quality screen; never changes frozen native acceptance."""
import json,time
from pathlib import Path
r=Path(__file__).resolve().parent
for p in sorted((r/'runs').glob('*/*/attempts/*/result.json')):
 if 'Pick' not in p.parent.parent.parent.name:continue
 result=json.loads(p.read_text());out=p.parent/'grasp_quality.json'
 if not result['acceptance']['success'] or out.exists():continue
 contacts=json.loads((p.parent/'contacts.json').read_text());window=contacts[-24:]
 detail=result['observation']['object_details'];target_bodies=set(detail.get('target',detail.get('articulated',{})).get('bodies',{}))
 hand_names={b for c in window for e in c['events'] if any(x in target_bodies for x in e['bodies']) for b in e['bodies'] if 'hand' in b and e['force_N']>.1}
 maintained=len(window)==24 and all(c['hand_object_N']>.1 for c in window)
 support_free=bool(window) and all(c['support_N']<.1 for c in window)
 frame_names=[{b for e in c['events'] if e['force_N']>.1 and any(x in target_bodies for x in e['bodies']) for b in e['bodies'] if 'hand' in b} for c in window]
 bilateral=any(k in p.parent.parent.parent.name for k in ['Box','Plate','Basket'])
 opposing_proxy=bool(frame_names) and all(len(names)>=2 and (not bilateral or all(any(side+'_hand' in b for b in names) for side in ['left','right'])) for names in frame_names)
 valid=maintained and support_free and opposing_proxy
 x={'native_success':True,'valid_grasp':valid,'training_ready':False,'measured_hand_contact_maintained_0_12s':maintained,'support_free_0_12s':support_free,'distinct_hand_contact_bodies':sorted(hand_names),'reason':'Supplementary screen: sustained hand contact, two hand parts and no support at native height pass.' if valid else 'Native height pass lacks sustained multi-part hand support evidence.','limitations':['This screen is not force closure proof.','Native execution stops at its height criterion; one-second post-lift hold has not been tested.'],'audited_unix_s':time.time()}
 out.write_text(json.dumps(x,indent=2));print(p,valid)
