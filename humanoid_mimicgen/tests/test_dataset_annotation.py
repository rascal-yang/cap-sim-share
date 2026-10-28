"""Input integrity and annotation boundary tests; no physics is advanced."""
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def demo_fixture():
    n = 6
    pose = np.tile([0., 0., 0., 1., 0., 0., 0.], (n, 1))
    return {
        "time_s": np.arange(n) * .02,
        **{key: pose.copy() for key in ["left_pose", "right_pose", "object_pose", "root_pose", "torso_pose"]},
        "left_fingers": np.zeros((n, 7)), "right_fingers": np.zeros((n, 7)),
        "hand_force_N": np.array([0., 0., 0., 3., 4., 4.]),
        "support_force_N": np.array([1., 1., 1., 0., 0., 0.]),
        "table_force_N": np.zeros(n),
        "metadata": {"object_id": "target", "finger_values": "measured_qpos_not_commands"},
    }


def annotation_fixture(demo):
    from humanoid_mimicgen.dataset import demo_digest
    return {"schema_version": 1, "demo_sha256": demo_digest(demo), "stages": [
        {"name": "approach", "kind": "transit", "start_frame": 0, "end_frame": 3,
         "hand": "right", "object_id": None, "evidence": ["No hand contact before frame 3."]},
        {"name": "grasp and lift", "kind": "contact", "start_frame": 3, "end_frame": 6,
         "hand": "right", "object_id": "target", "evidence": ["Force is 3 to 4 N on frames 3 through 5."]},
    ]}


def test_demo_roundtrip_detects_archive_or_metadata_tampering(tmp_path):
    from humanoid_mimicgen.dataset import save_demo, load_demo, demo_digest
    demo = demo_fixture()
    path = tmp_path / "demo.npz"
    save_demo(demo, path)
    loaded = load_demo(path)
    assert demo_digest(loaded) == demo_digest(demo)
    np.testing.assert_array_equal(loaded["hand_force_N"], [0., 0., 0., 3., 4., 4.])
    with pytest.raises(FileExistsError):
        save_demo(demo, path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    metadata["metadata"]["object_id"] = "other"
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="digest"):
        load_demo(path)
    archive = tmp_path / "archive.npz"
    save_demo(demo, archive)
    with archive.open("ab") as stream: stream.write(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        load_demo(archive)


@pytest.mark.parametrize("change", ["nan", "time", "shape", "quaternion", "missing_force"])
def test_demo_rejects_invalid_measured_arrays(change):
    from humanoid_mimicgen.dataset import validate_demo
    demo = demo_fixture()
    if change == "nan": demo["right_pose"][0, 0] = np.nan
    if change == "time": demo["time_s"][2] = demo["time_s"][1]
    if change == "shape": demo["right_fingers"] = np.zeros((5, 7))
    if change == "quaternion": demo["object_pose"][0, 3:] = 0
    if change == "missing_force": del demo["hand_force_N"]
    with pytest.raises(ValueError): validate_demo(demo)


def test_annotation_binds_complete_half_open_coverage_and_contact_evidence():
    from humanoid_mimicgen.annotation import validate_annotation, build_request
    demo = demo_fixture()
    annotation = annotation_fixture(demo)
    validate_annotation(annotation, demo)
    request = build_request(demo)
    assert request["frame_count"] == 6
    assert request["contact_intervals"] == [{"start_frame": 3, "end_frame": 6,
                                             "start_time_s": .06, "last_time_s": .1}]
    assert request["sampled_frames"][-1]["frame"] == 5
    broken = copy.deepcopy(annotation)
    broken["demo_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"): validate_annotation(broken, demo)
    for change in ["gap", "overlap", "last", "fractional", "empty_evidence", "no_contact", "unknown_object"]:
        broken = copy.deepcopy(annotation)
        if change == "gap": broken["stages"][1]["start_frame"] = 4
        if change == "overlap": broken["stages"][1]["start_frame"] = 2
        if change == "last": broken["stages"][1]["end_frame"] = 5
        if change == "fractional": broken["stages"][0]["start_frame"] = 0.0
        if change == "empty_evidence": broken["stages"][1]["evidence"] = []
        if change == "no_contact": broken["stages"][0].update(kind="contact", object_id="target")
        if change == "unknown_object": broken["stages"][1]["object_id"] = "other"
        with pytest.raises(ValueError): validate_annotation(broken, demo)


def test_annotation_cannot_discard_measured_contact_as_transit():
    from humanoid_mimicgen.annotation import validate_annotation
    demo = demo_fixture()
    annotation = annotation_fixture(demo)
    annotation["stages"][1].update(kind="transit", object_id=None)
    with pytest.raises(ValueError, match="contact"):
        validate_annotation(annotation, demo)


def _native_fixture(path):
    import mujoco
    from cap_sim.io import digest
    bodies = ['<body name="torso_link" pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/>']
    for side in ["left", "right"]:
        bodies.append(f'<body name="{side}_wrist_yaw_link" pos=".1 0 0"><geom type="sphere" size=".01" mass=".1"/>')
        for finger, count in [("thumb", 3), ("middle", 2), ("index", 2)]:
            for number in range(count):
                bodies.append(f'<body><joint name="{side}_hand_{finger}_{number}_joint" type="hinge"/><geom type="sphere" size=".01" mass=".1"/></body>')
        bodies.append('</body>')
    bodies.append('</body><body name="apple_001" pos=".5 0 .7"><freejoint/><geom type="sphere" size=".03" mass=".1"/></body>')
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody>' + ''.join(bodies) + '</worldbody></mujoco>')
    mujoco.mj_saveModel(model, str(path / "scene.mjb"), None)
    data = mujoco.MjData(model)
    rows = [np.r_[t, data.qpos, data.qvel, data.ctrl] for t in [.005, .01, .015]]
    np.savez_compressed(path / "states.npz", samples=rows)
    values = {
        "samples.json": [{"time": t, "dt": .005, "native_success": i == 2} for i, t in enumerate([.005, .01, .015])],
        "contacts.json": [{"hand_object_N": 2., "support_N": 0., "robot_table_N": 0.} for _ in rows],
        "result.json": {"acceptance": {"success": True, "sim_seconds": .015},
                        "observation": {"object_details": {"target": {"root_body": "apple_001"}}}},
        "independent_validation.json": {"success": True, "states_checked": 3, "physics_steps": 0},
    }
    for filename, value in values.items(): (path / filename).write_text(json.dumps(value))
    success = path.parent.parent / "success" / ("attempt_" + path.name)
    success.mkdir(parents=True)
    for name in ["states.npz", "scene.mjb", "contacts.json", "samples.json", "independent_validation.json"]:
        shutil.copyfile(path / name, success / name)
    (success / "trajectory.json").write_text(json.dumps({
        "source_attempt": str(path.resolve()), "samples_sha256": digest(values["samples.json"]),
        "acceptance": values["result.json"]["acceptance"],
    }))
    return success


def test_native_import_is_fk_only_and_requires_success_evidence(tmp_path, monkeypatch):
    import mujoco
    from humanoid_mimicgen.dataset import import_native
    source = tmp_path / "attempts/001"
    source.mkdir(parents=True)
    _native_fixture(source)
    def forbidden(*args, **kwargs): pytest.fail("Importer advanced physics")
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    demo = import_native(source, tmp_path / "import.npz")
    np.testing.assert_allclose(demo["object_pose"][0], [.5, 0, .7, 1, 0, 0, 0])
    assert demo["metadata"]["finger_values"] == "measured_qpos_not_commands"
    assert len(demo["metadata"]["source_sha256"]["states.npz"]) == 64
    (source / "independent_validation.json").write_text('{"success": false}')
    with pytest.raises(ValueError, match="independent"):
        import_native(source, tmp_path / "rejected.npz")


@pytest.mark.parametrize("changed", ["source_attempt", "samples_sha256", "acceptance", "success_copy", "missing_trajectory", "object_body"])
def test_native_import_binds_historical_success_to_exact_source_and_target(tmp_path, changed):
    from humanoid_mimicgen.dataset import import_native
    source = tmp_path / "attempts/001"
    source.mkdir(parents=True)
    success = _native_fixture(source)
    trajectory = success / "trajectory.json"
    record = json.loads(trajectory.read_text())
    if changed == "source_attempt": record[changed] = str(tmp_path / "other")
    if changed == "samples_sha256": record[changed] = "0" * 64
    if changed == "acceptance": record[changed]["sim_seconds"] = .03
    trajectory.write_text(json.dumps(record))
    if changed == "success_copy": (success / "contacts.json").write_text("[]")
    if changed == "missing_trajectory": trajectory.unlink()
    body = "torso_link" if changed == "object_body" else "apple_001"
    with pytest.raises(ValueError):
        import_native(source, tmp_path / "rejected.npz", object_body=body)
    assert not (tmp_path / "rejected.npz").exists()


def test_api_route_records_real_request_and_rejects_incomplete_response(tmp_path, monkeypatch):
    from humanoid_mimicgen.annotation import annotate_openai, build_request
    request = build_request(demo_fixture())
    monkeypatch.setenv("TEST_MIMIC_KEY", "test-key")
    response = {"id": "response-test", "model": "gpt-6-astra", "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(annotation_fixture(demo_fixture()))}]}]}
    class Reply:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(response).encode()
    def open_request(req, timeout):
        body = json.loads(req.data)
        assert body["model"] == "gpt-6-astra"
        assert body["text"]["format"]["strict"] is True
        return Reply()
    monkeypatch.setattr("urllib.request.urlopen", open_request)
    result = annotate_openai(request, "gpt-6-astra", tmp_path / "success", key_env="TEST_MIMIC_KEY")
    assert result["stages"][-1]["end_frame"] == 6
    assert json.loads((tmp_path / "success/timing.json").read_text())["status"] == "completed"
    assert "test-key" not in ''.join(p.read_text() for p in (tmp_path / "success").iterdir())
    response["status"] = "incomplete"
    with pytest.raises(RuntimeError, match="Incomplete"):
        annotate_openai(request, "gpt-6-astra", tmp_path / "failed", key_env="TEST_MIMIC_KEY")
    response.update(status="completed", model="different-model")
    with pytest.raises(RuntimeError, match="model"):
        annotate_openai(request, "gpt-6-astra", tmp_path / "substituted", key_env="TEST_MIMIC_KEY")
    with pytest.raises(ValueError, match="GPT-6"):
        annotate_openai(request, "another-model", tmp_path / "wrong", key_env="TEST_MIMIC_KEY")
