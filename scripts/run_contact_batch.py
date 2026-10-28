#!/usr/bin/env python3
"""System Python 3.11+ launcher; credentials never written to experiment artifacts."""
import argparse
import json
import os
from pathlib import Path
import sys
import tomllib


def configure_credentials(primary_path, secondary_path, provider):
    first=tomllib.loads(Path(primary_path).read_text())['model_providers']['OpenAI']
    if first['base_url'].rstrip('/') != provider['base_url'].rstrip('/'):
        raise ValueError('Primary endpoint does not match credential configuration')
    os.environ['AUGMENTATION_API_KEY']=first['experimental_bearer_token']
    os.environ['AUGMENTATION_ACTOR_HEADER']=first['http_headers']['x-openai-actor-authorization']
    if secondary_path is None:
        os.environ.pop('QUALITY_SUITE_FALLBACK_CONFIG',None)
        return None
    second=tomllib.loads(Path(secondary_path).read_text())['model_providers']['OpenAI']
    os.environ['AUGMENTATION_SECONDARY_API_KEY']=second['experimental_bearer_token']
    os.environ['AUGMENTATION_SECONDARY_ACTOR_HEADER']=second['http_headers']['x-openai-actor-authorization']
    fallback=provider|{'base_url':second['base_url'],'key_env':'AUGMENTATION_SECONDARY_API_KEY',
        'headers_env':{'x-openai-actor-authorization':'AUGMENTATION_SECONDARY_ACTOR_HEADER'}}
    os.environ['QUALITY_SUITE_FALLBACK_CONFIG']=json.dumps(fallback)
    return fallback


def configure_additional_credentials(paths, provider):
    configs=[]
    for i,path in enumerate(paths,start=3):
        entry=tomllib.loads(Path(path).read_text())['model_providers']['OpenAI']
        key=f'AUGMENTATION_API_{i}_KEY';actor=f'AUGMENTATION_API_{i}_ACTOR_HEADER'
        os.environ[key]=entry['experimental_bearer_token']
        os.environ[actor]=entry['http_headers']['x-openai-actor-authorization']
        configs.append(provider|{'base_url':entry['base_url'],'key_env':key,'headers_env':{'x-openai-actor-authorization':actor}})
    os.environ['QUALITY_SUITE_ADDITIONAL_CONFIGS']=json.dumps(configs)
    return configs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--provider',required=True)
    p.add_argument('--credentials',default='../api/openai_config.toml')
    p.add_argument('--secondary-credentials',help='Second credential configuration for alternating requests or failover')
    p.add_argument('--additional-credentials',nargs='*',default=[],help='Additional endpoint credential files, e.g. third API')
    p.add_argument('--api-routing',choices=('alternate','failover'),default='alternate')
    args=p.parse_args();repo=Path(__file__).resolve().parents[1]
    config=Path(args.config).resolve();provider=Path(args.provider).resolve()
    configure_credentials(args.credentials,args.secondary_credentials,json.loads(provider.read_text()))
    if args.additional_credentials and not args.secondary_credentials:p.error('Additional credentials require --secondary-credentials')
    configure_additional_credentials(args.additional_credentials,json.loads(provider.read_text()))
    os.environ['QUALITY_SUITE_API_ROUTING']=args.api_routing
    os.environ['PYTHONPATH']=str(repo)+os.pathsep+os.environ.get('PYTHONPATH','')
    os.chdir(repo)
    os.execv(str(repo/'scripts/humano_python.sh'),['humano_python.sh','-m','humanoid_mimicgen.augmentation.contact_batch','--config',str(config),'--provider',str(provider)])


if __name__=='__main__':main()
