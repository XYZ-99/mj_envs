import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
from mj_envs.utils.quatmath import *
import os
import mujoco_py

ADD_BONUS_REWARDS = True

class HammerEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.target_obj_sid = -1
        self.S_grasp_sid = -1
        self.obj_bid = -1
        self.tool_sid = -1
        self.goal_sid = -1
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_hammer.xml', 5)
        utils.EzPickle.__init__(self)

        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])
        
        self.target_obj_sid = self.sim.model.site_name2id('S_target')
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        self.tool_sid = self.sim.model.site_name2id('tool')
        self.goal_sid = self.sim.model.site_name2id('nail_goal')
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:,1])
        self.action_space.low  = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:,0])

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a * self.act_rng  # mean center and scale
        except:
            a = a  # only for the initialization phase
        a[5] = 0
        a[6] = 0
        a[7] = 0

#         qpos = [-1.08395366e-01, -8.61610663e-04, -1.74799641e-01, -5.17733132e-02,
#  -2.35464336e-04,  8.51813949e-03,  7.66294756e-03,  7.71110958e-03,
#   1.98648469e-04,  8.10292036e-01,  8.00413097e-01,  8.00432757e-01,
#   3.80358878e-03,  7.38452214e-01,  7.48875366e-01,  7.74608167e-01,
#   3.98107122e-01, -6.88627580e-03,  8.60622812e-01,  8.18404901e-01,
#   7.99360197e-01, -5.43019268e-03,  6.79864938e-01,  1.80767723e-02,
#   5.10429140e-03, -7.82594697e-01, -3.79621326e-21, -5.67412386e-04,
#  -4.73306750e-01, -1.52438697e-04,  9.47859512e-04, -1.02599300e-05,
#   2.76621992e-02, -9.75842984e-01,  3.45984479e-01,  4.14758417e-01,
#   2.20583496e-02, -4.16870713e-02, -1.00755453e-01]
        
#         qpos0 = [0 for i in range(38)]
#         qpos1 = [-1.02599300e-05, 2.76621992e-02, -9.75842984e-01,  3.45984479e-01,  4.14758417e-01,
#   2.20583496e-02, -4.16870713e-02, -1.00755453e-01]
#         qpos = qpos0 + qpos1

#         old_state = self.sim.get_state()
#         new_state = mujoco_py.MjSimState(old_state.time, qpos, old_state[2],
#                                          old_state.act, old_state.udd_state)
#         self.sim.set_state(new_state)
#         self.sim.forward()
        # print(a)
        # print(self.model.actuator_forcelimited)
        # print(self.model.actuator_gainprm[0])
        # print(self.model.actuator_biasprm[0])
        # print(self.data.actuator_length)
        # # print(self.model.actuator_lengthrange)
        # print(self.data.actuator_force)
        # print("\n\n")
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        tool_pos = self.data.site_xpos[self.tool_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        goal_pos = self.data.site_xpos[self.goal_sid].ravel()
        
        # get to hammer
        reward = - 0.1 * np.linalg.norm(palm_pos - obj_pos)
        # take hammer head to nail
        reward -= np.linalg.norm((tool_pos - target_pos))
        # make nail go inside
        reward -= 10 * np.linalg.norm(target_pos - goal_pos)
        # velocity penalty
        reward -= 1e-2 * np.linalg.norm(self.data.qvel.ravel())

        if ADD_BONUS_REWARDS:
            # bonus for lifting up the hammer
            if obj_pos[2] > 0.04 and tool_pos[2] > 0.04:
                reward += 2

            # bonus for hammering the nail
            if (np.linalg.norm(target_pos - goal_pos) < 0.020):
                reward += 25
            if (np.linalg.norm(target_pos - goal_pos) < 0.010):
                reward += 75

        goal_achieved = True if np.linalg.norm(target_pos - goal_pos) < 0.010 else False

        return ob, reward, False, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        qv = np.clip(self.data.qvel.ravel(), -1.0, 1.0)
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        obj_rot = quat2euler(self.data.body_xquat[self.obj_bid].ravel()).ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        nail_impact = np.clip(self.sim.data.sensordata[self.sim.model.sensor_name2id('S_nail')], -1.0, 1.0)
        return np.concatenate([qp[:-6], qv[-6:], palm_pos, obj_pos, obj_rot, target_pos, np.array([nail_impact])])

    def reset_model(self):
        self.sim.reset()
        target_bid = self.model.body_name2id('nail_board')
        self.model.body_pos[target_bid,2] = self.np_random.uniform(low=0.1, high=0.25)
        self.sim.forward()
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        board_pos = self.model.body_pos[self.model.body_name2id('nail_board')].copy()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel().copy()
        return dict(qpos=qpos, qvel=qvel, board_pos=board_pos, target_pos=target_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        board_pos = state_dict['board_pos']
        self.set_state(qp, qv)
        self.model.body_pos[self.model.body_name2id('nail_board')] = board_pos
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 45
        self.viewer.cam.distance = 2.0
        self.sim.forward()

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if nail insude board for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
