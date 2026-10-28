"""Lossless metadata encoding of legitimate infinity sentinels, never robot states."""
import argparse,copy,json,math
from pathlib import Path
from . import suite_physics as physics
from . import quality_suite as q
from .provider import ResponsesProvider
from .agent import save


def encode_infinities(value):
    if isinstance(value,dict):return {k:encode_infinities(v) for k,v in value.items()}
    if isinstance(value,list):return [encode_infinities(v) for v in value]
    if isinstance(value,float):
        if math.isnan(value):raise ValueError('NaN is not an unbounded metadata sentinel')
        if math.isinf(value):return {'ieee754':'+Infinity' if value>0 else '-Infinity'}
    return value


def install():
    if getattr(physics,'_explicit_infinity_encoding',False):return
    original_write=physics.write_new;original_metrics=physics.native_metrics
    def write(path,value):
        if Path(path).name=='physics.json':
            value=copy.deepcopy(value);value['jnt_range']=encode_infinities(value['jnt_range'])
        if Path(path).name=='experiment.json':
            value=copy.deepcopy(value);value['pipeline']['metadata_serialization']='suite_evidence: encode +/-Infinity joint range and native metric sentinels as explicit ieee754 objects; no state/control/physics change; NaN rejected'
        return original_write(path,value)
    def metrics(b,passed):return encode_infinities(original_metrics(b,passed))
    physics.write_new=write;physics.native_metrics=metrics;physics._explicit_infinity_encoding=True


def run(config,provider):
    cfg=json.loads(Path(config).read_text());root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    install();original_decide=q.api_decide
    def decide(p,r,context,episodes):
        return original_decide(p,r,{**context,'new_tool_diagnostic':cfg['diagnostic']},episodes)
    q.api_decide=decide
    result=q.run_task(Path(cfg['dataset']),root,ResponsesProvider(json.loads(Path(provider).read_text())),cfg['episodes'],3)
    save(root/'result.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--provider',required=True);a=p.parse_args();r=run(a.config,a.provider);print(json.dumps({'task':r['task'],'status':r['status']}))
