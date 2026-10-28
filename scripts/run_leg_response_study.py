#!/usr/bin/env python3
"""System-Python launcher; credentials stay in environment, never experiment files."""
import argparse,json,os
from pathlib import Path
from run_contact_batch import configure_credentials,configure_additional_credentials
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--provider',required=True);p.add_argument('--credentials',required=True);p.add_argument('--secondary-credentials',required=True);p.add_argument('--additional-credentials',nargs='*',default=[]);a=p.parse_args()
provider=json.loads(Path(a.provider).read_text());configure_credentials(a.credentials,a.secondary_credentials,provider);configure_additional_credentials(a.additional_credentials,provider);os.environ['QUALITY_SUITE_API_ROUTING']='alternate'
repo=Path(__file__).resolve().parents[1];os.environ['PYTHONPATH']=str(repo)
os.execv(str(repo/'scripts/humano_python.sh'),['humano_python.sh','-c','import json,sys; from humanoid_mimicgen.augmentation.leg_response_study import run; run(json.load(open(sys.argv[1])),json.load(open(sys.argv[2])))',str(Path(a.config).resolve()),str(Path(a.provider).resolve())])
