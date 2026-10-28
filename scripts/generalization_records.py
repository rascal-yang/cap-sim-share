"""Resolve condition results across explicit recovery runs without editing evidence."""
import json
from pathlib import Path

def roots(root):
    root=Path(root);out=[root];p=root/'continuations.json'
    if p.exists():
        for child in json.loads(p.read_text()):out.extend(roots(child))
    return list(dict.fromkeys(out))

def conditions(root):
    out={}
    for r in roots(root):
        for p in sorted(r.glob('*_??/summary.json')):
            v=json.loads(p.read_text());v['_summary_path']=str(p);key=v['condition_id']
            if key in out and out[key]['status']=='complete':raise ValueError('Recovery overwrites a complete condition: '+key)
            out[key]=v
    return [out[k] for k in sorted(out)]
