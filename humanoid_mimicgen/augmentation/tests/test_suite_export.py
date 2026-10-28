from humanoid_mimicgen.augmentation.suite_export import case_grade


def test_native_success_is_not_api_quality_acceptance():
    assert case_grade({'success':True,'mode':'cartesian_source_legs','object_delta_xy_m':[.01,0]},None)=='awaiting_api_review'
    assert case_grade({'success':True,'mode':'cartesian_source_legs','object_delta_xy_m':[.01,0]},'reject')=='api_rejected'
    assert case_grade({'success':False},'accept')=='physical_failure'
    assert case_grade({'success':True},'accept')=='api_accepted_physical'
