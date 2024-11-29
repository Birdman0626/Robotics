from dofbot import DofbotEnv
import numpy as np
import time

if __name__ == '__main__':
    env = DofbotEnv()
    env.reset()
    Reward = False

    '''
    constants here
    '''
    GRIPPER_DEFAULT_ANGLE = 20. / 180. * 3.1415
    GRIPPER_CLOSE_ANGLE = -20. / 180. * 3.1415

    # define state machine
    INITIAL_STATE = 0
    GRASP_STATE = 1 
    LIFT_STATE = 2
    PUT_STATE = 3
    MOVE_STATE = 4
    BACK_STATE = 5
    current_state = INITIAL_STATE

    initial_jointposes = [1.57, 0., 1.57, 1.57, 1.57]

    # offset to grasp object
    obj_offset = [-0.023, -0.023, 0.09]
    obj_offset2 = [-0.032, 0.032, 0.13]
    obj_offset3 = [-0.025, 0.025, 0.09]

    block_pos, block_orn = env.get_block_pose()

    start_time = None

    while not Reward:
        '''
        #获取物块位姿、目标位置和机械臂位姿，计算机器臂关节和夹爪角度，使得机械臂夹取绿色物块，放置到紫色区域。
        '''

        '''
        code here
        '''

        if current_state == INITIAL_STATE:
            #把机械臂挪过去
            grasp_pos = (block_pos[0] + obj_offset[0], block_pos[1] + obj_offset[1], block_pos[2] + obj_offset[2])
            grasp_joint_pos = env.dofbot_setInverseKine(grasp_pos, -1 * block_orn)
            env.dofbot_control(grasp_joint_pos, GRIPPER_DEFAULT_ANGLE)
            current_joint_pos, _ = env.get_dofbot_jointPoses()
            if np.all(np.isclose(np.array(current_joint_pos), np.array(grasp_joint_pos), atol = 1e-2)):
                current_state = GRASP_STATE
                start_time = time.time()
         
        elif current_state == GRASP_STATE:
            #抓两秒
            env.dofbot_control(grasp_joint_pos, GRIPPER_CLOSE_ANGLE)               
            current_time = time.time()
            if (current_time - start_time > 2.):
                current_state = LIFT_STATE
                lift_pos = (block_pos[0] + obj_offset[0], block_pos[1] + obj_offset[1],
                            block_pos[2] + obj_offset[2] + 0.02)
                
        elif current_state == LIFT_STATE:
            #稍微抬一下
            lift_joint_pos = env.dofbot_setInverseKine(lift_pos, -1 * block_orn)
            env.dofbot_control(lift_joint_pos, GRIPPER_CLOSE_ANGLE)
            current_joint_pos, _ = env.get_dofbot_jointPoses()
            if np.all(np.isclose(np.array(current_joint_pos), np.array(lift_joint_pos), atol = 1e-2)):
                current_state = MOVE_STATE
                target_pos = env.get_target_pose()
                move_pos = (target_pos[0] + obj_offset2[0], target_pos[1] + obj_offset2[1], block_pos[2] + obj_offset2[2])
                
        elif current_state == MOVE_STATE:
            #挪过去
            move_joint_pos = env.dofbot_setInverseKine(move_pos, block_orn * -1)
            env.dofbot_control(move_joint_pos, GRIPPER_CLOSE_ANGLE)
            current_joint_pos, _ = env.get_dofbot_jointPoses()
            if np.all(np.isclose(np.array(current_joint_pos), np.array(move_joint_pos), atol = 1e-2)):
                current_state = BACK_STATE
                back_pos = (target_pos[0] + obj_offset3[0], target_pos[1] + obj_offset3[1], block_pos[2] + obj_offset3[2])
                
        elif current_state == BACK_STATE:
            #再挪过去然后松开
            back_joint_pos = env.dofbot_setInverseKine(back_pos, block_orn * -1)
            current_joint_pos, _ = env.get_dofbot_jointPoses()
            if np.all(np.isclose(np.array(current_joint_pos), np.array(back_joint_pos), atol = 1e-2)):
                env.dofbot_control(back_joint_pos, GRIPPER_DEFAULT_ANGLE)
            else:
                env.dofbot_control(back_joint_pos, GRIPPER_CLOSE_ANGLE)

        Reward = env.reward()
