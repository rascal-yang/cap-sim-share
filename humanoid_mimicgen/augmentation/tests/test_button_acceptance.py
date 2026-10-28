from humanoid_mimicgen.augmentation.button_acceptance import accepts_contact,hold_result

def test_exact_palm_mesh_only_not_entire_wrist():
 assert accepts_contact('right_hand_index_0_link','index',revised=False)
 assert not accepts_contact('right_wrist_yaw_link','right_hand_palm_link',revised=False)
 assert accepts_contact('right_wrist_yaw_link','right_hand_palm_link',revised=True)
 assert not accepts_contact('right_wrist_yaw_link','right_wrist_yaw_link',revised=True)
 assert not accepts_contact('right_wrist_pitch_link','right_hand_palm_link',revised=True)

def test_same_force_distance_count_and_continuous_duration_gates():
 valid={'force_N':20.,'distance_m':.001,'contacts':1}
 assert hold_result([valid]*50)['success']
 assert not hold_result([valid]*49)['success']
 assert not hold_result([valid]*30+[dict(valid,force_N=19.99)]+[valid]*30)['success']
 assert not hold_result([dict(valid,distance_m=.00101)]*60)['success']
 assert not hold_result([dict(valid,contacts=0)]*60)['success']
