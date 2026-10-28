"""Object-anchored trajectory retargeting, without simulation or success claims.

Poses use world xyz followed by quaternion wxyz. Adjacent contact phases share
the object pose at their FIRST sample as a fixed anchor. The original action
sequence remains available; this module does not generate a new task strategy.
"""

from copy import deepcopy
import hashlib
import json

import numpy as np

from .annotation import validate_annotation
from .dataset import ARRAY_KEYS, FINGER_KEYS, FORCE_KEYS, POSE_KEYS, demo_digest, validate_demo


def _parameters(translation, rotation_wxyz, preserve_prefix_s):
    try:
        translation = np.asarray(translation, dtype=float)
        rotation = np.asarray(rotation_wxyz, dtype=float)
        prefix = float(preserve_prefix_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("Transform parameters must be finite numeric values") from exc
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("translation must contain three finite coordinates")
    if rotation.shape != (4,) or not np.isfinite(rotation).all():
        raise ValueError("rotation_wxyz must contain four finite coordinates")
    norm = np.linalg.norm(rotation)
    if not np.isclose(norm, 1.0, atol=1e-5, rtol=0):
        raise ValueError("rotation_wxyz must be a unit quaternion")
    if not np.isfinite(prefix) or prefix < 0:
        raise ValueError("preserve_prefix_s must be finite and nonnegative")
    return translation, rotation / norm, prefix


def _annotation_digest(annotation):
    encoded = json.dumps(annotation, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _hands(stage):
    return ("left", "right") if stage["hand"] == "both" else (stage["hand"],)


def _contact_groups(annotation):
    groups = []
    for stage in annotation["stages"]:
        if stage["kind"] != "contact":
            continue
        if groups and groups[-1]["end_frame"] == stage["start_frame"]:
            groups[-1]["end_frame"] = stage["end_frame"]
            groups[-1]["stages"].append(stage)
        else:
            groups.append({"start_frame": stage["start_frame"],
                           "end_frame": stage["end_frame"], "stages": [stage]})
    return groups


def _quat_multiply(left, right):
    lw, lx, ly, lz = np.moveaxis(np.asarray(left), -1, 0)
    rw, rx, ry, rz = np.moveaxis(np.asarray(right), -1, 0)
    result = np.stack((lw * rw - lx * rx - ly * ry - lz * rz,
                       lw * rx + lx * rw + ly * rz - lz * ry,
                       lw * ry - lx * rz + ly * rw + lz * rx,
                       lw * rz + lx * ry - ly * rx + lz * rw), axis=-1)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _rotation_matrix(quaternion):
    w, x, y, z = quaternion
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _transform(poses, anchor, translation, rotation):
    """T_new_anchor @ inverse(T_source_anchor) @ T_source_pose.

    Since q_new_anchor = rotation * q_source_anchor, its world rotation delta
    is exactly rotation; the source anchor orientation cancels algebraically.
    """
    poses = np.asarray(poses)
    result = np.empty(poses.shape, dtype=float)
    result[..., :3] = ((poses[..., :3] - anchor[:3]) @ _rotation_matrix(rotation).T
                       + anchor[:3] + translation)
    result[..., 3:] = _quat_multiply(rotation, poses[..., 3:])
    return result


def _slerp(start, end, weights):
    start = np.asarray(start, dtype=float) / np.linalg.norm(start)
    end = np.asarray(end, dtype=float) / np.linalg.norm(end)
    dot = float(np.dot(start, end))
    if dot < 0:
        end, dot = -end, -dot
    weights = np.asarray(weights)[:, None]
    if dot > 0.9995:
        result = (1 - weights) * start + weights * end
    else:
        angle = np.arccos(np.clip(dot, -1, 1))
        result = (np.sin((1 - weights) * angle) * start
                  + np.sin(weights * angle) * end) / np.sin(angle)
    return result / np.linalg.norm(result, axis=1, keepdims=True)


def _active_masks(annotation, n):
    active = {hand: np.zeros(n, dtype=bool) for hand in ("left", "right")}
    transit = {hand: np.zeros(n, dtype=bool) for hand in ("left", "right")}
    for stage in annotation["stages"]:
        segment = slice(stage["start_frame"], stage["end_frame"])
        for hand in _hands(stage):
            active[hand][segment] = True
            if stage["kind"] == "transit":
                transit[hand][segment] = True
    return active, transit


def _connect_transits(generated, annotation, prefix_count):
    times = np.asarray(generated["time_s"])
    n = len(times)
    _, transit = _active_masks(annotation, n)
    for hand, mask in transit.items():
        changes = np.diff(np.r_[False, mask, False].astype(int))
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            start = max(int(start), prefix_count)
            if start >= end:
                continue
            # These endpoints lie outside the modified samples whenever possible.
            # The last original sample is the endpoint for a trailing transit.
            left, right = max(start - 1, 0), min(int(end), n - 1)
            if right == left:
                continue
            poses = generated[f"{hand}_pose"]
            initial, target = poses[left].copy(), poses[right].copy()
            u = (times[start:end] - times[left]) / (times[right] - times[left])
            weight = u * u * u * (10 + u * (-15 + 6 * u))
            poses[start:end, :3] = initial[:3] + weight[:, None] * (target[:3] - initial[:3])
            poses[start:end, 3:] = _slerp(initial[3:], target[3:], weight)


def _expected_objects(demo, groups, translation, rotation):
    """Apply fixed anchors to the reference, including the initial scene target.

    Before the first contact use its anchor; between contacts retain the
    preceding group's anchor until the next group starts. These are expected
    object states, never observations or commands to a physical object.
    """
    expected = np.empty_like(demo["object_pose"], dtype=float)
    for index, group in enumerate(groups):
        start = group["start_frame"] if index else 0
        end = groups[index + 1]["start_frame"] if index + 1 < len(groups) else len(expected)
        anchor = np.asarray(demo["object_pose"])[group["start_frame"]]
        expected[start:end] = _transform(np.asarray(demo["object_pose"])[start:end],
                                         anchor, translation, rotation)
    return expected


def retarget_demo(demo, annotation, translation=(0, 0, 0),
                  rotation_wxyz=(1, 0, 0, 0), preserve_prefix_s=5.0):
    """Return independent arrays with fixed-anchor contact reuse and transits.

    ``preserve_prefix_s`` is an ABSOLUTE source simulation timestamp, inclusive;
    it is not a duration relative to the first recorded sample. Both wrist pose
    commands in that prefix remain exact. A changed contact overlapping it is
    rejected rather than silently violating either invariant.

    The exact identity transform preserves every original array, including its
    transit trajectory. Otherwise active transit hands use quintic Cartesian
    interpolation and shortest-arc SLERP, bounded by adjacent preserved/contact
    samples. Root, torso, fingers, sample times and inactive hands never change.
    Forces remain SOURCE evidence and object poses are EXPECTATIONS only.
    """
    validate_annotation(annotation, demo)
    translation, rotation, prefix = _parameters(translation, rotation_wxyz, preserve_prefix_s)
    groups = _contact_groups(annotation)
    if not groups:
        raise ValueError("Contact retargeting requires at least one contact stage")
    times = np.asarray(demo["time_s"])
    prefix_count = int(np.searchsorted(times, prefix, side="right"))
    identity = bool(np.all(translation == 0) and np.all(rotation[1:] == 0))
    if not identity and any(group["start_frame"] < prefix_count for group in groups):
        raise ValueError("Changed contact overlaps retained prefix; shorten the prefix explicitly")
    generated = deepcopy(demo)
    if not identity:
        for key in POSE_KEYS:
            generated[key] = np.array(demo[key], dtype=float, copy=True)
        for group in groups:
            anchor = np.asarray(demo["object_pose"])[group["start_frame"]]
            for stage in group["stages"]:
                segment = slice(stage["start_frame"], stage["end_frame"])
                for hand in _hands(stage):
                    key = f"{hand}_pose"
                    generated[key][segment] = _transform(np.asarray(demo[key])[segment], anchor,
                                                         translation, rotation)
        generated["object_pose"] = _expected_objects(demo, groups, translation, rotation)
        _connect_transits(generated, annotation, prefix_count)
    generated["metadata"]["retargeting"] = {
        "schema_version": 1, "method": "fixed-object-anchor-contact-trajectory-retargeting",
        "research_kind": "trajectory_retargeting", "original_sequences_accessible": True,
        "source_demo_sha256": demo_digest(demo), "annotation_sha256": _annotation_digest(annotation),
        "translation_m": translation.tolist(), "rotation_wxyz": rotation.tolist(),
        "preserve_prefix_s": prefix, "prefix_time_basis": "absolute_source_simulation_time_inclusive",
        "preserved_prefix_frames": prefix_count, "zero_transform_identity": identity,
        "contact_group_anchors": [group["start_frame"] for group in groups],
        "transit_method": "source_identity" if identity else "quintic_xyz_shortest_arc_slerp",
        "trailing_transit_endpoint": "final_source_pose",
        "object_pose_semantics": "expected-only",
        "object_transit_anchor_rule": "preceding_group_or_first_group_before_first_contact",
        "force_data_semantics": "source-demonstration-only",
        "force_data_fields": list(FORCE_KEYS), "force_data_is_execution_evidence": False,
        "finger_values": "preserved_source_measurements_not_original_commands",
        "physical_success_verified": False, "physics_steps": 0,
    }
    validate_demo(generated)
    return generated


def _pose_errors(expected, actual):
    expected, actual = np.asarray(expected), np.asarray(actual)
    position = np.linalg.norm(expected[..., :3] - actual[..., :3], axis=-1)
    qa = expected[..., 3:] / np.linalg.norm(expected[..., 3:], axis=-1, keepdims=True)
    qb = actual[..., 3:] / np.linalg.norm(actual[..., 3:], axis=-1, keepdims=True)
    angle = 2 * np.arccos(np.clip(np.abs(np.sum(qa * qb, axis=-1)), 0, 1))
    return position, angle


def geometry_metrics(demo, generated, annotation):
    """Check reference geometry and invariants, never physical feasibility.

    Boundary values are measured adjacent-sample steps, NOT a promise of zero
    velocity/acceleration discontinuity or collision-free robot execution. Stage
    hand changes with no intervening transit can expose large steps here.
    """
    validate_annotation(annotation, demo)
    validate_demo(generated)
    n = len(demo["time_s"])
    if len(generated["time_s"]) != n:
        raise ValueError("Generated and source sample counts differ")
    info = generated["metadata"].get("retargeting")
    if not isinstance(info, dict):
        raise ValueError("Generated trajectory lacks retargeting provenance")
    translation, rotation, prefix = _parameters(info.get("translation_m"),
                                                info.get("rotation_wxyz"),
                                                info.get("preserve_prefix_s"))
    groups = _contact_groups(annotation)
    if not groups:
        raise ValueError("Contact metrics require at least one contact stage")
    prefix_count = int(np.searchsorted(np.asarray(demo["time_s"]), prefix, side="right"))
    active, _ = _active_masks(annotation, n)
    metrics = {"evaluation": "offline_reference_geometry_only", "physics_steps": 0,
               "physical_success_verified": False, "contact_groups": [], "boundary_steps": []}
    maximum_position, maximum_angle = 0., 0.
    for group in groups:
        anchor_frame = group["start_frame"]
        anchor = np.asarray(demo["object_pose"])[anchor_frame]
        group_position, group_angle = 0., 0.
        for stage in group["stages"]:
            segment = slice(stage["start_frame"], stage["end_frame"])
            for hand in _hands(stage):
                key = f"{hand}_pose"
                expected = _transform(np.asarray(demo[key])[segment], anchor, translation, rotation)
                position, angle = _pose_errors(expected, np.asarray(generated[key])[segment])
                group_position = max(group_position, float(np.max(position)))
                group_angle = max(group_angle, float(np.max(angle)))
        metrics["contact_groups"].append({
            "start_frame": group["start_frame"], "end_frame": group["end_frame"],
            "anchor_frame": anchor_frame, "phase_names": [s["name"] for s in group["stages"]],
            "position_error_max_m": group_position, "orientation_error_max_rad": group_angle})
        maximum_position, maximum_angle = max(maximum_position, group_position), max(maximum_angle, group_angle)
    object_position, object_angle = _pose_errors(_expected_objects(demo, groups, translation, rotation),
                                                generated["object_pose"])
    metrics.update(contact_position_error_max_m=maximum_position,
                   contact_orientation_error_max_rad=maximum_angle,
                   object_reference_position_error_max_m=float(np.max(object_position)),
                   object_reference_orientation_error_max_rad=float(np.max(object_angle)))
    boundaries = {stage["start_frame"] for stage in annotation["stages"][1:]}
    if 0 < prefix_count < n:
        boundaries.add(prefix_count)
    for frame in sorted(boundaries):
        row = {"frame": frame, "time_s": float(demo["time_s"][frame]),
               "sample_dt_s": float(demo["time_s"][frame] - demo["time_s"][frame - 1])}
        for hand in ("left", "right"):
            key = f"{hand}_pose"
            step, angle = _pose_errors(np.asarray(generated[key])[frame - 1], generated[key][frame])
            source_step, source_angle = _pose_errors(np.asarray(demo[key])[frame - 1], demo[key][frame])
            row.update({f"{hand}_position_step_m": float(step),
                        f"{hand}_orientation_step_rad": float(angle),
                        f"{hand}_source_position_step_m": float(source_step),
                        f"{hand}_source_orientation_step_rad": float(source_angle)})
        metrics["boundary_steps"].append(row)
    equal = lambda key, mask=slice(None): np.array_equal(np.asarray(demo[key])[mask],
                                                       np.asarray(generated[key])[mask])
    flags = {
        "time_preserved": equal("time_s"),
        "fingers_preserved": all(equal(key) for key in FINGER_KEYS),
        "body_preserved": all(equal(key) for key in ("root_pose", "torso_pose")),
        "inactive_hands_preserved": all(equal(f"{hand}_pose", ~mask) for hand, mask in active.items()),
        "prefix_preserved": all(equal(key, slice(0, prefix_count)) for key in
                                ("left_pose", "right_pose", "root_pose", "torso_pose") + FINGER_KEYS),
        "source_forces_preserved": all(equal(key) for key in FORCE_KEYS),
        "source_identity_matches": info.get("source_demo_sha256") == demo_digest(demo),
        "annotation_identity_matches": info.get("annotation_sha256") == _annotation_digest(annotation),
        "reference_semantics_declared": (info.get("object_pose_semantics") == "expected-only"
                                         and info.get("force_data_semantics") == "source-demonstration-only"
                                         and info.get("physical_success_verified") is False),
    }
    if info.get("zero_transform_identity"):
        flags["identity_reference_preserved"] = all(equal(key) for key in ARRAY_KEYS)
    metrics.update(flags)
    metrics["invariants_passed"] = bool(all(flags.values()) and maximum_position < 1e-9
                                        and maximum_angle < 1e-6 and np.max(object_position) < 1e-9
                                        and np.max(object_angle) < 1e-6)
    return metrics
