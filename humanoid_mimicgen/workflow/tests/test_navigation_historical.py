"""Optional local integration fixtures reference real frozen archives, never rewrite them."""
import json
from pathlib import Path
import pytest
from humanoid_mimicgen.workflow.navigation.__main__ import extract
from humanoid_mimicgen.workflow.navigation.prior import transfer_stance

ROOT=Path(__file__).parent/'fixtures/navigation'

@pytest.mark.parametrize('task,expected',[('apple','completed'),('plate','unsupported_entry_pattern'),('button','completed'),('chair','completed')])
def test_four_real_archive_adapters(task,expected):
    manifest=ROOT/(task+'.json');spec=json.loads(manifest.read_text())
    if not all(Path(path).is_file() for path in spec['files'].values()):
        pytest.skip('Frozen historical archive is not installed on this machine')
    prior=extract(manifest,ROOT/'extraction_options.json')
    assert prior['status']==expected
    assert prior['physics_steps']==0
    assert prior['support_evidence_kind']=='geometric_contact_not_force'
    if task=='button':assert prior['success_criterion']=='button_palm_mesh_v1'
    if expected=='completed':
        frame=prior['source_task_frame_xyzyaw'].copy();frame[0]+=.02
        assert transfer_stance(prior,frame)[0]==pytest.approx(prior['source_stance_xyyaw'][0]+.02)
