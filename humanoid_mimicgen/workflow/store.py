"""Content-addressed, atomic strategy packages with copied immutable evidence."""
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from .contracts import WorkflowError, canonical, digest, file_hash, validate_strategy


def safe_name(name):
    if not isinstance(name,str) or not name or '\\' in name:
        raise WorkflowError('invalid_input','Unsafe artifact name')
    p=PurePosixPath(name)
    if p.is_absolute() or any(part in ('..','.') for part in name.split('/')) or '' in name.split('/'):
        raise WorkflowError('invalid_input','Unsafe artifact name')
    return name


class Store:
    def __init__(self,root):
        self.root=Path(root).resolve()

    def save(self,strategy,artifacts):
        validate_strategy(strategy)
        self.root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.building-',dir=self.root) as tmp:
            staging=Path(tmp);hashes={}
            for name,source in sorted(artifacts.items()):
                name=safe_name(name);source=Path(source)
                if not source.is_file():raise WorkflowError('invalid_input','Missing artifact '+name)
                target=staging/'artifacts'/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target);hashes[name]=file_hash(target)
            manifest={'strategy':strategy,'artifacts':hashes};ident=digest(manifest)
            (staging/'manifest.json').write_bytes(canonical(manifest))
            target=self.root/ident
            try:os.rename(staging,target)
            except OSError:
                if not target.is_dir():raise
                self.load(ident)
            return ident

    def load(self,ident):
        if not isinstance(ident,str) or not re.fullmatch('[0-9a-f]{64}',ident):
            raise WorkflowError('invalid_input','Invalid strategy ID')
        p=self.root/ident
        if p.is_symlink() or not p.is_dir():raise WorkflowError('integrity_error','Missing strategy package')
        try:
            manifest=json.loads((p/'manifest.json').read_text())
            if digest(manifest)!=ident:raise WorkflowError('integrity_error','Manifest digest changed')
            validate_strategy(manifest['strategy'])
            for name,sha in manifest['artifacts'].items():
                target=p/'artifacts'/safe_name(name)
                if any(x.is_symlink() for x in [target,*target.parents] if x!=p.parent) or not target.is_file() or file_hash(target)!=sha:
                    raise WorkflowError('integrity_error','Artifact changed: '+name)
        except (OSError,KeyError,json.JSONDecodeError) as exc:
            raise WorkflowError('integrity_error','Unreadable package') from exc
        return {'strategy':manifest['strategy'],'manifest':manifest,'path':p,'strategy_id':ident}
