#!/usr/bin/env python3
"""Load explicitly configured API credentials into environment, then pinned HUMANO."""
import argparse
import json
import os
from pathlib import Path
from run_contact_batch import configure_credentials
import tomllib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', required=True)
parser.add_argument('--provider', required=True)
parser.add_argument('--credentials', default='../api/openai_config.toml')
parser.add_argument('--credentials-pool', nargs='+', help='Rotate these credential files on every request, including retries')
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
config = Path(args.config).resolve()
provider = Path(args.provider).resolve()
provider_config = json.loads(provider.read_text())
if args.credentials_pool:
    configure_credentials(args.credentials_pool[0], None, provider_config)
    profiles = []
    for index, name in enumerate(args.credentials_pool):
        path = Path(name)
        entry = tomllib.loads(path.read_text())['model_providers']['OpenAI']
        key = f'LATERAL_POOL_{index}_KEY'
        actor = f'LATERAL_POOL_{index}_ACTOR'
        os.environ[key] = entry['experimental_bearer_token']
        os.environ[actor] = entry['http_headers']['x-openai-actor-authorization']
        profiles.append(dict(label=path.name, config=provider_config | dict(base_url=entry['base_url'], key_env=key,
                             headers_env={'x-openai-actor-authorization': actor})))
    os.environ['LATERAL_API_POOL_CONFIG'] = json.dumps(profiles)
    manifest = Path(json.loads(config.read_text())['output_root'])/'api_pool.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != profiles:
            raise ValueError('API pool differs from frozen manifest')
    else:
        manifest.write_text(json.dumps(profiles, indent=2)+'\n')
else:
    os.environ.pop('LATERAL_API_POOL_CONFIG', None)
    configure_credentials(args.credentials, None, provider_config)
os.environ['PYTHONPATH'] = str(repo) + os.pathsep + os.environ.get('PYTHONPATH', '')
os.chdir(repo)
os.execv(str(repo / 'scripts/humano_python.sh'), ['humano_python.sh', '-m',
         'humanoid_mimicgen.augmentation.lateral_agent', '--config', str(config), '--provider', str(provider)])
