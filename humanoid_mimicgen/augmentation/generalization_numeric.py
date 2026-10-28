"""Numeric validation feedback only; never changes or clips API decisions."""
import math
from pathlib import Path
from . import generalization_benchmark as benchmark
from . import rollback_benchmark as rb

ORIGINAL_VALIDATE=rb.validate_candidates
SKILL=Path(__file__).parent/'skills/scene_generalization_numeric/SKILL.md'

def validate_candidates(decision,spec):
    try:return ORIGINAL_VALIDATE(decision,spec)
    except ValueError as error:
        details=[]
        for i,candidate in enumerate(decision.get('candidates',[])):
            p=candidate.get('parameters',{})
            if 'wrist_delta_m' in p:
                v=p['wrist_delta_m']
                if isinstance(v,list) and len(v)==3 and all(type(x) in (int,float) and math.isfinite(x) for x in v):
                    norm=math.sqrt(sum(x*x for x in v))
                    if norm>.05+1e-12:details.append(f'candidate[{i}] norm={norm:.9f}m >0.05m. Constraint is x*x+y*y+z*z<=0.0025; z=0.05 permits ONLY x=y=0. Reduce z before adding XY; do not resubmit the invalid vector.')
            tail=p.get('tail_s')
            if type(tail) in (int,float) and math.isfinite(tail) and abs(tail/.02-round(tail/.02))>1e-6:
                details.append(f'candidate[{i}] tail={tail}s is {tail/.02} ticks, not an integer. Select an integer tick count then multiply by0.02. For example1.24/1.26 are valid,1.25 is invalid;1.74/1.76 valid,1.75 invalid.')
        raise ValueError(str(error)+'; '+'; '.join(details)) from None

def run(config,provider_config):
    old_skill=benchmark.SKILL;old_validator=rb.validate_candidates
    benchmark.SKILL=SKILL;rb.validate_candidates=validate_candidates
    try:return benchmark.run(config,provider_config)
    finally:benchmark.SKILL=old_skill;rb.validate_candidates=old_validator
