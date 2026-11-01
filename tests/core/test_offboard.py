import math

import numpy as np

from quadsim.deploy.offboard import OffboardPolicy

CONTRACT = {
    "observation": {"layout": [["lin_vel_b_est", 3, ""], ["rot_mat_cols01", 6, ""], ["gyro_b", 3, ""],
                               ["last_action", 4, ""], ["task", 5, ""]]},
    "action": {"dim": 4, "control_mode": "ctbr", "collective_range": [0.0, 1.0], "max_total_thrust_N": 34.19432,
               "max_body_rate_rad_s": [6.0, 6.0, 3.0], "max_velocity_m_s": [3, 3, 2], "max_yaw_rate_rad_s": 2.0,
               "hover_action": [0.1844574511051178, 0.0, 0.0, 0.0]},
}
LEVEL = np.array([1.0, 0.0, 0.0, 0.0])


class FakePolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.obs = []

    def __call__(self, obs):
        self.obs.append(obs.copy())
        return self.action[None]


def odom(pos=(0.0, 0.0, 2.0)):
    return np.asarray(pos, dtype=np.float64), LEVEL, np.zeros(3), np.zeros(3)


def make(action=(0.3, 0.5, -0.5, 1.0), **kw):
    pol = FakePolicy(action)
    return OffboardPolicy(CONTRACT, pol, goal=(0.0, 0.0, 2.0), **kw), pol


def test_heartbeat_requires_fresh_odometry_inside_fence():
    c, pol = make()
    assert c.step(None, 0.0, True).reason == "no_odometry"
    for age in (0.051, float("nan"), float("inf")):
        r = c.step(odom(), age, True)
        assert (r.heartbeat, r.setpoint, r.reason) == (False, None, "stale_odometry")
    r = c.step(odom((3.5, 0.0, 2.0)), 0.01, True)  # 3.5 m from the goal, fence 3 m
    assert (r.heartbeat, r.setpoint, r.reason) == (False, None, "geofence")
    r = c.step(odom((float("nan"), 0.0, 2.0)), 0.01, True)
    assert not r.heartbeat
    assert pol.obs == []  # the policy never ran


def test_not_offboard_heartbeat_only_and_last_action_is_hover():
    c, pol = make()
    c.obs_builder.update_last_action(np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32))
    r = c.step(odom(), 0.01, offboard=False)
    assert (r.heartbeat, r.setpoint, r.action, r.reason) == (True, None, None, "not_offboard")
    assert np.allclose(c.obs_builder.last_action, CONTRACT["action"]["hover_action"])
    # first offboard step sees the hover action as "last action" (training episode start), then its own output
    c.step(odom(), 0.01, offboard=True)
    c.step(odom(), 0.01, offboard=True)
    la = slice(12, 16)
    assert np.allclose(pol.obs[0][0, la], CONTRACT["action"]["hover_action"])
    assert np.allclose(pol.obs[1][0, la], [0.3, 0.5, -0.5, 1.0])


def test_offboard_setpoint_and_thrust_scale():
    c, pol = make(action=(0.1844574511051178, 0.5, -0.5, 1.0), thrust_scale=1.2324)
    r = c.step(odom(), 0.01, offboard=True)
    assert r.heartbeat and r.reason == "ok"
    sp = r.setpoint
    # FLU rates (3, -3, 3) rad/s -> FRD (3, 3, -3); hover collective (m g / T_max = 0.592) x thrust_scale
    assert np.allclose([sp["roll"], sp["pitch"], sp["yaw"]], [3.0, 3.0, -3.0])
    assert math.isclose(-sp["thrust_body"][2], 0.592228725 * 1.2324, rel_tol=1e-6)
    assert c.heartbeat_flags() == {"body_rate": True, "velocity": False}


def test_policy_output_clipped_and_goal_in_observation():
    c, pol = make(action=(5.0, -5.0, 0.0, 0.0))
    c.goal = np.array([1.0, 0.0, 2.0])  # runtime goal change (ROS parameter callback)
    r = c.step(odom(), 0.0, offboard=True)
    assert np.allclose(r.action, [1.0, -1.0, 0.0, 0.0])
    assert np.allclose(pol.obs[0][0, 16:19], [1.0, 0.0, 0.0])  # target in body frame (level, yaw 0)


def test_log_row_follows_log_columns():
    from quadsim.deploy.offboard import LOG_COLUMNS

    c, _ = make()
    pos, q, vel, gyro = np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5, 0.5, 0.5]), np.array([4.0, 5, 6]), \
        np.array([7.0, 8, 9])
    row = dict(zip(LOG_COLUMNS, c.log_row(10.0, 9.9, 9.95, (pos, q, vel, gyro), np.array([0.1, 0.2, 0.3, 0.4]))))
    assert len(row) == len(LOG_COLUMNS) == len(c.log_row(0, 0, 0, (pos, q, vel, gyro), np.zeros(4)))
    assert [row[k] for k in ("px", "pz", "vx", "vz", "qw", "qz", "wx", "wz", "goal_z", "a0", "a3")] == \
        [1.0, 3.0, 4.0, 6.0, 0.5, 0.5, 7.0, 9.0, 2.0, 0.1, 0.4]
