"""Canonical measured demonstrations and physics-free native trace import."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

POSE_KEYS = ("left_pose", "right_pose", "object_pose", "root_pose", "torso_pose")
FINGER_KEYS = ("left_fingers", "right_fingers")
FORCE_KEYS = ("hand_force_N", "support_force_N", "table_force_N")
ARRAY_KEYS = ("time_s",) + POSE_KEYS + FINGER_KEYS + FORCE_KEYS


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_demo(demo):
    """Reject missing, nonfinite, misaligned or ambiguous measured arrays."""
    if not isinstance(demo, dict) or not isinstance(demo.get("metadata"), dict):
        raise ValueError("A demonstration needs arrays and metadata")
    if any(key not in demo for key in ARRAY_KEYS):
        raise ValueError("Missing demonstration array; absent force is unknown, not zero")
    arrays = {key: np.asarray(demo[key]) for key in ARRAY_KEYS}
    time_s = arrays["time_s"]
    if time_s.ndim != 1 or len(time_s) < 2:
        raise ValueError("time_s must contain at least two samples")
    n = len(time_s)
    for key, value in arrays.items():
        expected = (n, 7) if key in POSE_KEYS + FINGER_KEYS else (n,)
        if value.shape != expected or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"Invalid finite numeric shape for {key}: expected {expected}")
    if time_s[0] < 0 or np.any(np.diff(time_s) <= 0):
        raise ValueError("Sample times must be nonnegative and strictly increasing")
    for key in POSE_KEYS:
        if not np.allclose(np.linalg.norm(arrays[key][:, 3:], axis=1), 1., atol=1e-5, rtol=0):
            raise ValueError(f"{key} requires unit wxyz quaternions")
    if any(np.any(arrays[key] < 0) for key in FORCE_KEYS):
        raise ValueError("Contact force magnitudes cannot be negative")
    try:
        _json(demo["metadata"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Metadata must be finite JSON data") from exc


def demo_digest(demo):
    """Bind normalized float64 arrays and provenance, independent of NPZ compression."""
    validate_demo(demo)
    digest = hashlib.sha256()
    digest.update(_json(demo["metadata"]).encode())
    for key in ARRAY_KEYS:
        value = np.ascontiguousarray(demo[key], dtype="<f8")
        digest.update(_json([key, value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def save_demo(demo, path):
    """Save NPZ and same-stem JSON without replacing an existing artifact."""
    validate_demo(demo)
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Demonstration path must end in .npz")
    meta_path = path.with_suffix(".json")
    if path.exists() or meta_path.exists():
        raise FileExistsError(f"Demonstration artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **{key: np.asarray(demo[key], dtype="<f8") for key in ARRAY_KEYS})
    metadata = {"schema_version": 1, "metadata": demo["metadata"],
                "demo_sha256": demo_digest(demo), "npz_sha256": _sha256(path)}
    with meta_path.open("x") as stream:
        stream.write(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return path


def load_demo(path):
    path = Path(path)
    record = json.loads(path.with_suffix(".json").read_text())
    if record.get("schema_version") != 1 or record.get("npz_sha256") != _sha256(path):
        raise ValueError("Demonstration archive digest or schema mismatch")
    with np.load(path, allow_pickle=False) as data:
        demo = {key: data[key].copy() for key in data.files}
    demo["metadata"] = record["metadata"]
    validate_demo(demo)
    if record.get("demo_sha256") != demo_digest(demo):
        raise ValueError("Demonstration content digest mismatch")
    return demo


def import_native(source_dir, out_path, object_body="apple_001"):
    """Extract measured SE(3) poses using mj_kinematics, never mj_step.

    Acceptance records are checked and hashed as historical evidence. This does
    not rerun a task checker, recreate successful actions, or simulate a rollout.
    """
    import mujoco
    from cap_sim.io import digest as native_digest

    source_dir = Path(source_dir).resolve()
    required = ("states.npz", "scene.mjb", "samples.json", "contacts.json",
                "result.json", "independent_validation.json")
    names = list(required) + [name for name in ("controller.json", "program.json", "reset.json")
                              if (source_dir / name).is_file()]
    for name in required:
        if not (source_dir / name).is_file():
            raise ValueError(f"Required source evidence missing: {name}")
    hashes = {name: _sha256(source_dir / name) for name in names}
    read = lambda name: json.loads((source_dir / name).read_text())
    result = read("result.json")
    independent = read("independent_validation.json")
    if result.get("acceptance", {}).get("success") is not True:
        raise ValueError("Source native acceptance is not successful")
    if independent.get("success") is not True or independent.get("physics_steps") != 0:
        raise ValueError("Successful independent saved-state evidence is required")
    target_body = result.get("observation", {}).get("object_details", {}).get("target", {}).get("root_body")
    if not isinstance(target_body, str) or object_body != target_body:
        raise ValueError("Requested object_body must match the source acceptance target root_body")
    if source_dir.parent.name != "attempts" or not source_dir.name.isdigit():
        raise ValueError("Source must identify a native attempts/<number> directory")
    success_dir = source_dir.parent.parent / "success" / ("attempt_" + source_dir.name)
    trajectory_path = success_dir / "trajectory.json"
    if not trajectory_path.is_file():
        raise ValueError("Corresponding successful trajectory.json evidence is missing")
    trajectory_hash = _sha256(trajectory_path)
    trajectory = json.loads(trajectory_path.read_text())
    source_attempt = trajectory.get("source_attempt")
    if not isinstance(source_attempt, str) or Path(source_attempt).resolve() != source_dir:
        raise ValueError("Successful trajectory source_attempt differs from the imported source")
    samples, contacts = read("samples.json"), read("contacts.json")
    if trajectory.get("samples_sha256") != native_digest(samples):
        raise ValueError("Successful trajectory samples_sha256 does not match original samples")
    if trajectory.get("acceptance") != result["acceptance"]:
        raise ValueError("Successful trajectory acceptance differs from source result")
    success_files = ("states.npz", "scene.mjb", "contacts.json", "samples.json", "independent_validation.json")
    success_hashes = {}
    for name in success_files:
        path = success_dir / name
        if not path.is_file() or _sha256(path) != hashes[name]:
            raise ValueError("Successful trajectory copy differs from source evidence: " + name)
        success_hashes[name] = hashes[name]
    model = mujoco.MjModel.from_binary_path(str(source_dir / "scene.mjb"))
    data = mujoco.MjData(model)
    with np.load(source_dir / "states.npz", allow_pickle=False) as archive:
        states = archive["samples"].copy()
    width = 1 + model.nq + model.nv + model.nu
    if states.ndim != 2 or states.shape[1] != width or len(states) < 2 or not np.isfinite(states).all():
        raise ValueError("Recorded states shape or finiteness differs from the source model")
    n = len(states)
    if len(samples) != n or len(contacts) != n or independent.get("states_checked") != n:
        raise ValueError("Source states, samples, contacts and independent evidence are not aligned")
    if not np.allclose([row["time"] for row in samples], states[:, 0], atol=1e-9, rtol=0):
        raise ValueError("Source sample timestamps do not match states")
    if not any(row.get("native_success") is True for row in samples):
        raise ValueError("Source samples lack recorded native success")
    if not np.isclose(result["acceptance"].get("sim_seconds", np.nan), states[-1, 0], atol=1e-7, rtol=0):
        raise ValueError("Source acceptance end time differs from measured states")
    body_names = {"left_pose": "left_wrist_yaw_link", "right_pose": "right_wrist_yaw_link",
                  "torso_pose": "torso_link", "object_pose": object_body}
    body_ids = {key: model.body(name).id for key, name in body_names.items()}
    finger_names = {side: [f"{side}_hand_{finger}_{number}_joint"
                          for finger, count in (("thumb", 3), ("middle", 2), ("index", 2))
                          for number in range(count)] for side in ("left", "right")}
    finger_ids = {side: [int(model.joint(name).qposadr[0]) for name in names]
                  for side, names in finger_names.items()}
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_qposadr[0] != 0:
        raise ValueError("Native trace requires its robot root free joint first")
    demo = {key: np.empty((n, 7)) for key in POSE_KEYS + FINGER_KEYS}
    demo["time_s"] = states[:, 0].copy()
    for i, row in enumerate(states):
        data.qpos[:] = row[1:1 + model.nq]
        mujoco.mj_kinematics(model, data)
        for key, body_id in body_ids.items():
            demo[key][i] = np.r_[data.xpos[body_id], data.xquat[body_id]]
        demo["root_pose"][i] = data.qpos[:7]
        for side, addresses in finger_ids.items():
            demo[f"{side}_fingers"][i] = data.qpos[addresses]
    for key, native_key in zip(FORCE_KEYS, ("hand_object_N", "support_N", "robot_table_N")):
        if any(native_key not in row for row in contacts):
            raise ValueError(f"Missing measured contact field {native_key}")
        demo[key] = np.asarray([row[native_key] for row in contacts], dtype=float)
    modes = {}
    if "controller.json" in names:
        for row in read("controller.json"):
            mode = str(row.get("mode", "unknown"))
            modes[mode] = modes.get(mode, 0) + 1
    if hashes != {name: _sha256(source_dir / name) for name in names}:
        raise ValueError("Source evidence changed while importing")
    if (trajectory_hash != _sha256(trajectory_path)
            or success_hashes != {name: _sha256(success_dir / name) for name in success_files}):
        raise ValueError("Successful trajectory evidence changed while importing")
    demo["metadata"] = {
        "schema_version": 1, "object_id": "target", "object_body": object_body,
        "source_dir": str(source_dir), "source_sha256": hashes,
        "success_trajectory_path": str(trajectory_path), "success_trajectory_sha256": trajectory_hash,
        "success_copy_sha256": success_hashes,
        "imported_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "source_acceptance": result["acceptance"], "independent_validation": independent,
        "source_controller_modes": modes, "finger_values": "measured_qpos_not_commands",
        "finger_joint_names": finger_names, "pose_convention": "world_xyz_quaternion_wxyz",
        "import_method": "MuJoCo mj_kinematics on recorded qpos; recorded contact forces",
        "physics_steps": 0, "source_state_layout": {"nq": model.nq, "nv": model.nv, "nu": model.nu},
        "research_kind": "trajectory_retargeting", "original_sequences_accessible": True,
    }
    save_demo(demo, out_path)
    return demo
