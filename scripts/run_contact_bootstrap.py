#!/usr/bin/env python3
"""System Python 3.11+ launcher; credentials never written to experiment artifacts."""
import argparse
import os
from pathlib import Path
import sys
import tomllib


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--provider',required=True)
    p.add_argument('--credentials',default='../api/openai_config.toml')
    args=p.parse_args();repo=Path(__file__).resolve().parents[1]
    config=Path(args.config).resolve();provider=Path(args.provider).resolve()
    credentials=tomllib.loads(Path(args.credentials).read_text())['model_providers']['OpenAI']
    os.environ['AUGMENTATION_API_KEY']=credentials['experimental_bearer_token']
    os.environ['AUGMENTATION_ACTOR_HEADER']=credentials['http_headers']['x-openai-actor-authorization']
    os.environ['PYTHONPATH']=str(repo)+os.pathsep+os.environ.get('PYTHONPATH','')
    os.chdir(repo)
    os.execv(str(repo/'scripts/humano_python.sh'),['humano_python.sh','-m','humanoid_mimicgen.augmentation.bootstrap_agent','--config',str(config),'--provider',str(provider)])


if __name__=='__main__':main()
