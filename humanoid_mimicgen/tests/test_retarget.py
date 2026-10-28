"""Hand-computable checks for contact reuse, with no simulator dependency."""

from copy import deepcopy

import numpy as np
import pytest

from humanoid_mimicgen.dataset import demo_digest
from humanoid_mimicgen.retarget import geometry_metrics, retarget_demo


def demonstration():
    n = 15
    pose = np.tile([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (n, 1))
    demo = {
        "time_s": np.arange(n, dtype=float),
        "left_pose": pose.copy(),
        "right_pose": pose.copy(),
        "object_pose": pose.copy(),
        "root_pose": pose.copy(),
        "torso_pose": pose.copy(),
        "left_fingers": np.arange(n * 7, dtype=float).reshape(n, 7) / 100,
        "right_fingers": np.arange(n * 7, dtype=float).reshape(n, 7) / 200,
        "hand_force_N": np.zeros(n),
        "support_force_N": np.full(n, 100.0),
        "table_force_N": np.full(n, 1.0),
        "metadata": {},
    }
    demo["right_pose"][:, :3] = [2, 2, 3]
    demo["left_pose"][:, :3] = [-1, 2, 3]
    demo["object_pose"][:, :3] = [1, 2, 3]
    demo["root_pose"][:, :3] = [0, 0, 1]
    demo["torso_pose"][:, :3] = [0, 0, 1.5]
    demo["right_pose"][4:9, 2] += [0, 0.2, 0.4, 0.6, 0.8]
    demo["object_pose"][4:9, 2] += [0, 0.2, 0.4, 0.6, 0.8]
    demo["object_pose"][9:, 2] = 3.8
    demo["hand_force_N"][4:9] = 2.0
    return demo


def annotation(demo):
    def stage(name, kind, start, end):
        return {
            "name": name, "kind": kind, "start_frame": start,
            "end_frame": end, "hand": "right",
            "object_id": "target" if kind == "contact" else None,
            "evidence": ["hand_force_N positive in this interval" if kind == "contact"
                         else "hand_force_N zero in this interval"],
        }
    return {
        "schema_version": 1,
        "demo_sha256": demo_digest(demo),
        "stages": [stage("approach", "transit", 0, 4),
                   stage("grasp", "contact", 4, 7),
                   stage("lift", "contact", 7, 9),
                   stage("depart", "transit", 9, 15)],
    }


def test_translation_moves_every_contact_sample_and_expected_object():
    demo = demonstration()
    source = deepcopy(demo)
    generated = retarget_demo(demo, annotation(demo), translation=(0.4, -0.2, 0.5),
                              preserve_prefix_s=0)
    np.testing.assert_allclose(generated["right_pose"][4, :3], [2.4, 1.8, 3.5])
    np.testing.assert_allclose(generated["right_pose"][8, :3], [2.4, 1.8, 4.3])
    np.testing.assert_allclose(generated["object_pose"][0, :3], [1.4, 1.8, 3.5])
    np.testing.assert_allclose(generated["object_pose"][8, :3], [1.4, 1.8, 4.3])
    for key in demo:
        if key != "metadata":
            np.testing.assert_array_equal(demo[key], source[key])


def test_rotation_is_left_multiplied_around_object_anchor_not_world_origin():
    demo = demonstration()
    q = np.sqrt(0.5)
    generated = retarget_demo(demo, annotation(demo), translation=(0.5, 0, 0),
                              rotation_wxyz=(q, 0, 0, q), preserve_prefix_s=0)
    np.testing.assert_allclose(generated["right_pose"][4, :3], [1.5, 3, 3], atol=1e-12)
    np.testing.assert_allclose(generated["right_pose"][4, 3:], [q, 0, 0, q], atol=1e-12)
    np.testing.assert_allclose(generated["object_pose"][4, :3], [1.5, 2, 3], atol=1e-12)


def test_consecutive_contact_phases_share_one_fixed_anchor_and_preserve_lift():
    demo = demonstration()
    q = np.sqrt(0.5)
    generated = retarget_demo(demo, annotation(demo), rotation_wxyz=(q, q, 0, 0),
                              preserve_prefix_s=0)
    np.testing.assert_allclose(generated["right_pose"][7, :3], [2, 1.4, 3], atol=1e-12)
    np.testing.assert_allclose(generated["right_pose"][8, :3], [2, 1.2, 3], atol=1e-12)
    np.testing.assert_allclose(generated["object_pose"][8, :3], [1, 1.2, 3], atol=1e-12)
    metrics = geometry_metrics(demo, generated, annotation(demo))
    assert len(metrics["contact_groups"]) == 1
    assert metrics["contact_groups"][0]["anchor_frame"] == 4
    assert metrics["contact_position_error_max_m"] < 1e-12
    assert metrics["contact_orientation_error_max_rad"] < 1e-7


def test_inactive_hand_body_fingers_and_times_are_exactly_preserved():
    demo = demonstration()
    generated = retarget_demo(demo, annotation(demo), translation=(0.1, 0, 0),
                              preserve_prefix_s=0)
    for key in ("left_pose", "root_pose", "torso_pose", "left_fingers",
                "right_fingers", "time_s"):
        np.testing.assert_array_equal(generated[key], demo[key])
    provenance = generated["metadata"]["retargeting"]
    assert provenance["object_pose_semantics"] == "expected-only"
    assert provenance["force_data_semantics"] == "source-demonstration-only"
    assert provenance["physical_success_verified"] is False
    for key in ("hand_force_N", "support_force_N", "table_force_N"):
        np.testing.assert_array_equal(generated[key], demo[key])


def test_newly_active_left_hand_uses_same_contact_group_anchor():
    demo = demonstration()
    stages = annotation(demo)
    stages["stages"][2]["hand"] = "both"
    q = np.sqrt(0.5)
    generated = retarget_demo(demo, stages, rotation_wxyz=(q, q, 0, 0),
                              preserve_prefix_s=0)
    np.testing.assert_array_equal(generated["left_pose"][:7], demo["left_pose"][:7])
    np.testing.assert_array_equal(generated["left_pose"][9:], demo["left_pose"][9:])
    np.testing.assert_allclose(generated["left_pose"][7, :3], [-1, 2, 3], atol=1e-12)
    np.testing.assert_allclose(generated["left_pose"][7, 3:], [q, q, 0, 0], atol=1e-12)


def test_transit_uses_samples_outside_segment_for_quintic_endpoints():
    demo = demonstration()
    generated = retarget_demo(demo, annotation(demo), translation=(1, 0, 0),
                              preserve_prefix_s=0)
    # The boundary endpoints are retained frame 0 and contact frame 4.
    np.testing.assert_allclose(generated["right_pose"][:5, 0],
                               [2, 2.103515625, 2.5, 2.896484375, 3])
    # A trailing transit returns smoothly to the final original pose.
    np.testing.assert_allclose(generated["right_pose"][-1], demo["right_pose"][-1])
    metrics = geometry_metrics(demo, generated, annotation(demo))
    entering = [b for b in metrics["boundary_steps"] if b["frame"] == 4][0]
    assert entering["right_position_step_m"] == pytest.approx(0.103515625)
    assert metrics["time_preserved"]
    assert metrics["fingers_preserved"]
    assert metrics["prefix_preserved"]


def test_prefix_uses_absolute_simulation_time_including_nonzero_first_timestamp():
    demo = demonstration()
    demo["time_s"] = np.arange(15, dtype=float) + 0.005
    # Delay contact until after the default five-second prefix.
    demo["hand_force_N"][:] = 0
    demo["hand_force_N"][8:12] = 2
    ann = annotation(demo)
    ann["stages"][0]["end_frame"] = 8
    ann["stages"][1].update(start_frame=8, end_frame=10)
    ann["stages"][2].update(start_frame=10, end_frame=12)
    ann["stages"][3]["start_frame"] = 12
    generated = retarget_demo(demo, ann, translation=(1, 0, 0))
    np.testing.assert_array_equal(generated["right_pose"][:5], demo["right_pose"][:5])
    assert generated["right_pose"][5, 0] > demo["right_pose"][5, 0]
    assert generated["metadata"]["retargeting"]["preserved_prefix_frames"] == 5


def test_quaternion_sign_equivalence_never_introduces_a_long_arc():
    demo = demonstration()
    demo["right_pose"][4:, 3:] *= -1
    q = np.sqrt(0.5)
    generated = retarget_demo(demo, annotation(demo), rotation_wxyz=(q, 0, 0, q),
                              preserve_prefix_s=0)
    np.testing.assert_allclose(np.abs(generated["right_pose"][2, 3:]),
                               [np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)], atol=1e-12)
    generated["right_pose"][4:9, 3:] *= -1
    metrics = geometry_metrics(demo, generated, annotation(demo))
    assert metrics["contact_orientation_error_max_rad"] < 1e-7


def test_metrics_detect_corrupted_contact_and_preserved_finger_samples():
    demo = demonstration()
    generated = retarget_demo(demo, annotation(demo), translation=(0.2, 0, 0),
                              preserve_prefix_s=0)
    generated["right_pose"][7, 0] += 0.03
    generated["left_fingers"][2, 3] += 0.1
    metrics = geometry_metrics(demo, generated, annotation(demo))
    assert metrics["contact_position_error_max_m"] == pytest.approx(0.03)
    assert not metrics["fingers_preserved"]
    assert not metrics["invariants_passed"]


@pytest.mark.parametrize("rotation", [(1, 0, 0, 0), (-1, 0, 0, 0)])
def test_zero_transform_preserves_entire_reference_exactly(rotation):
    demo = demonstration()
    demo["right_pose"][::2, 3:] *= -1
    generated = retarget_demo(demo, annotation(demo), rotation_wxyz=rotation)
    for key in demo:
        if key != "metadata":
            np.testing.assert_array_equal(generated[key], demo[key])
    assert generated["metadata"]["retargeting"]["zero_transform_identity"]
    assert geometry_metrics(demo, generated, annotation(demo))["invariants_passed"]


def test_contact_covering_last_sample_is_transformed_without_index_overrun():
    demo = demonstration()
    demo["hand_force_N"][9:] = 2
    ann = annotation(demo)
    ann["stages"].pop()
    ann["stages"][-1]["end_frame"] = len(demo["time_s"])
    generated = retarget_demo(demo, ann, translation=(0.3, 0, 0), preserve_prefix_s=0)
    np.testing.assert_allclose(generated["right_pose"][-1, :3], [2.3, 2, 3])
    assert geometry_metrics(demo, generated, ann)["invariants_passed"]


def test_separate_contact_groups_reanchor_and_connect_across_the_gap():
    demo = demonstration()
    demo["hand_force_N"][7:] = 0
    demo["hand_force_N"][10:12] = 2
    ann = annotation(demo)
    ann["stages"][2] = dict(ann["stages"][0], name="reapproach", start_frame=7, end_frame=10)
    ann["stages"].insert(3, dict(ann["stages"][1], name="regrasp", start_frame=10, end_frame=12))
    ann["stages"][-1]["start_frame"] = 12
    q = np.sqrt(0.5)
    generated = retarget_demo(demo, ann, rotation_wxyz=(q, q, 0, 0), preserve_prefix_s=0)
    np.testing.assert_allclose(generated["right_pose"][6, :3], [2, 1.6, 3], atol=1e-12)
    np.testing.assert_allclose(generated["right_pose"][10, :3], [2, 2.8, 3.8], atol=1e-12)
    np.testing.assert_allclose(generated["right_pose"][8, :3], [2, 2.2, 3.4], atol=1e-12)
    metrics = geometry_metrics(demo, generated, ann)
    assert [group["anchor_frame"] for group in metrics["contact_groups"]] == [4, 10]
    assert metrics["invariants_passed"]


def test_noncommuting_source_orientation_is_left_multiplied():
    demo = demonstration()
    q = np.sqrt(0.5)
    demo["right_pose"][:, 3:] = [q, q, 0, 0]
    demo["object_pose"][:, 3:] = [0, 0, 1, 0]
    generated = retarget_demo(demo, annotation(demo), rotation_wxyz=(q, 0, 0, q),
                              preserve_prefix_s=0)
    np.testing.assert_allclose(generated["right_pose"][4, 3:], [0.5, 0.5, 0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(generated["object_pose"][4, 3:], [0, -q, q, 0], atol=1e-12)
    np.testing.assert_allclose(generated["right_pose"][4, :3], [1, 3, 3], atol=1e-12)


def test_a_transit_only_annotation_cannot_claim_contact_retargeting():
    demo = demonstration()
    ann = annotation(demo)
    ann["stages"] = [dict(ann["stages"][0], end_frame=15)]
    with pytest.raises(ValueError, match="contact"):
        retarget_demo(demo, ann)


def test_metrics_detect_object_reference_or_source_binding_corruption():
    demo = demonstration()
    generated = retarget_demo(demo, annotation(demo), translation=(0.3, 0, 0),
                              preserve_prefix_s=0)
    generated["object_pose"][5, 2] += 0.08
    generated["metadata"]["retargeting"]["source_demo_sha256"] = "0" * 64
    metrics = geometry_metrics(demo, generated, annotation(demo))
    assert metrics["object_reference_position_error_max_m"] == pytest.approx(0.08)
    assert not metrics["source_identity_matches"]
    assert not metrics["invariants_passed"]


def test_metrics_report_inactive_hand_or_prefix_corruption():
    demo = demonstration()
    generated = retarget_demo(demo, annotation(demo), translation=(0.3, 0, 0),
                              preserve_prefix_s=0)
    generated["left_pose"][5, 0] += 0.1
    generated["right_pose"][0, 0] += 0.1
    metrics = geometry_metrics(demo, generated, annotation(demo))
    assert not metrics["inactive_hands_preserved"]
    assert not metrics["prefix_preserved"]
    assert not metrics["invariants_passed"]


@pytest.mark.parametrize("arguments", [
    {"translation": (1, 2)}, {"translation": (0, np.nan, 0)},
    {"rotation_wxyz": (0, 0, 0, 0)}, {"rotation_wxyz": (1, 0, 0)},
    {"rotation_wxyz": (np.inf, 0, 0, 0)},
    {"preserve_prefix_s": -1}, {"preserve_prefix_s": np.nan},
])
def test_invalid_transform_arguments_are_rejected(arguments):
    demo = demonstration()
    with pytest.raises(ValueError):
        retarget_demo(demo, annotation(demo), **arguments)


def test_malformed_demo_and_annotation_are_rejected():
    demo = demonstration()
    ann = annotation(demo)
    bad_demo = deepcopy(demo)
    bad_demo["right_pose"][5, 0] = np.nan
    with pytest.raises(ValueError):
        retarget_demo(bad_demo, ann, preserve_prefix_s=0)
    ann["stages"][1]["start_frame"] = 5
    with pytest.raises(ValueError):
        retarget_demo(demo, ann, preserve_prefix_s=0)


def test_changed_contact_overlapping_retained_prefix_is_rejected():
    demo = demonstration()
    with pytest.raises(ValueError, match="prefix"):
        retarget_demo(demo, annotation(demo), translation=(0.1, 0, 0))
