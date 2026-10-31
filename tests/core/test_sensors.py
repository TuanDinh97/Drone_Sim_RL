import math

import pytest
import torch

from quadsim.core import math as qm
from quadsim.core.disturbances import WindCfg, WindModel
from quadsim.core.sensors import (
    BarometerModel,
    BaroCfg,
    DelayBuffer,
    DepthNoiseCfg,
    GaussMarkov,
    ImuModel,
    ImuNoiseCfg,
    SampleAndHold,
    apply_depth_noise,
    cast,
    pinhole_ray_dirs,
)


def test_delay_buffer():
    buf = DelayBuffer(3, max_delay=4, feature_shape=(1,), device="cpu")
    buf.set_delay(torch.tensor([0, 2, 4]))
    buf.reset(torch.arange(3), torch.full((3, 1), -1.0))
    outs = [buf.push(torch.full((3, 1), float(k))) for k in range(8)]
    assert [o[0, 0].item() for o in outs] == list(range(8))
    assert [o[1, 0].item() for o in outs] == [-1, -1] + list(range(6))
    assert [o[2, 0].item() for o in outs] == [-1] * 4 + list(range(4))


def test_sample_and_hold_rate():
    sh = SampleAndHold(1, (1,), rate_hz=50.0, sim_dt=0.005, device="cpu")
    vals = [sh.step(torch.tensor([[float(k)]]))[0].item() for k in range(12)]
    assert vals == [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8]


def test_gauss_markov_stationary_std():
    torch.manual_seed(0)
    gm = GaussMarkov((20000, 1), sigma=0.3, tau=0.5, device="cpu")
    gm.reset()
    for _ in range(200):
        gm.step(0.01)
    assert gm.x.std().item() == pytest.approx(0.3, rel=0.03)


def test_random_walk_growth():
    torch.manual_seed(1)
    gm = GaussMarkov((20000, 1), 0.0, None, "cpu", random_walk_sigma=0.1)
    gm.reset(initial_std=0.0)
    for _ in range(400):
        gm.step(0.01)
    assert gm.x.std().item() == pytest.approx(0.1 * math.sqrt(4.0), rel=0.03)


def _imu(cfg, n=4000):
    return ImuModel(cfg, n, sim_dt=0.005, device="cpu")


def test_imu_specific_force_level_and_tilted():
    cfg = ImuNoiseCfg(enabled=True, gyro_noise_density=0, gyro_random_walk=0, gyro_turn_on_bias=0,
                      acc_noise_density=0, acc_random_walk=0, acc_turn_on_bias=0)
    imu = _imu(cfg, 2)
    q = torch.stack([torch.tensor([1.0, 0, 0, 0]), qm.quat_from_euler_xyz(*[torch.tensor(v) for v in (0.0, 0.5, 0.0)])])
    imu.reset(torch.arange(2), torch.zeros(2, 3))
    out = None
    for _ in range(2):
        out = imu.step(q, torch.zeros(2, 3), torch.zeros(2, 3))
    expected = qm.quat_rotate_inverse(q, torch.tensor([0.0, 0.0, 9.81]).expand(2, 3))
    assert torch.allclose(out[:, 3:], expected, atol=1e-5)
    # free fall measures zero specific force
    imu.reset(torch.arange(2), torch.zeros(2, 3))
    v = torch.zeros(2, 3)
    imu.step(q, v, torch.zeros(2, 3))
    v = v + torch.tensor([0.0, 0.0, -9.81 * 0.005])
    out = imu.step(q, v, torch.zeros(2, 3))
    assert out[:, 3:].abs().max().item() < 1e-4


def test_imu_white_noise_density():
    torch.manual_seed(2)
    cfg = ImuNoiseCfg(rate_hz=200.0, gyro_random_walk=0, gyro_turn_on_bias=0, acc_random_walk=0, acc_turn_on_bias=0)
    imu = _imu(cfg)
    n = imu.n
    imu.reset(torch.arange(n), torch.zeros(n, 3))
    q = torch.tensor([1.0, 0, 0, 0]).expand(n, 4)
    out = imu.step(q, torch.zeros(n, 3), torch.zeros(n, 3))
    g_std = out[:, :3].std().item()
    a_std = (out[:, 3:] - torch.tensor([0, 0, 9.81])).std().item()
    assert g_std == pytest.approx(cfg.gyro_noise_density * math.sqrt(200.0), rel=0.05)
    assert a_std == pytest.approx(cfg.acc_noise_density * math.sqrt(200.0), rel=0.05)


def test_imu_turn_on_bias_and_latency():
    torch.manual_seed(3)
    cfg = ImuNoiseCfg(gyro_noise_density=0, gyro_random_walk=0, acc_noise_density=0, acc_random_walk=0,
                      delay_steps_max=0)
    imu = _imu(cfg)
    n = imu.n
    imu.reset(torch.arange(n), torch.zeros(n, 3))
    out = imu.step(torch.tensor([1.0, 0, 0, 0]).expand(n, 4), torch.zeros(n, 3), torch.zeros(n, 3))
    assert out[:, :3].std().item() == pytest.approx(cfg.gyro_turn_on_bias, rel=0.05)
    # latency: a step change in angular velocity appears exactly `delay` steps later
    cfg2 = ImuNoiseCfg(enabled=True, gyro_noise_density=0, gyro_random_walk=0, gyro_turn_on_bias=0, acc_noise_density=0,
                       acc_random_walk=0, acc_turn_on_bias=0, delay_steps_max=3, rate_hz=200.0)
    imu2 = _imu(cfg2, 1)
    imu2.reset(torch.arange(1), torch.zeros(1, 3))
    imu2.delay.set_delay(torch.tensor([3]))
    q = torch.tensor([[1.0, 0, 0, 0]])
    seq = [imu2.step(q, torch.zeros(1, 3), torch.tensor([[float(k >= 2), 0.0, 0.0]]))[0, 0].item() for k in range(8)]
    assert seq == [0, 0, 0, 0, 0, 1, 1, 1]


def test_barometer_isa():
    p = BarometerModel.pressure_from_altitude(torch.tensor(100.0))
    assert p.item() == pytest.approx(100129.4, abs=1.0)
    h = BarometerModel.altitude_from_pressure(p)
    assert h.item() == pytest.approx(100.0, abs=1e-2)
    baro = BarometerModel(BaroCfg(noise_std=0.0, bias_std=0.0), 2, 0.005, "cpu")
    baro.reset(torch.arange(2))
    assert torch.allclose(baro.step(torch.tensor([1.5, 2.0]))[:, 0], torch.tensor([1.5, 2.0]))


def test_raycast_geometry():
    o = torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 5.0], [5.0, 0.0, 1.0]]])
    d = torch.tensor([[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [1.0, 0.0, -1.0], [1.0, 0.0, -0.6], [1.0, 0.0, 0.0]]])
    d = d / d.norm(dim=-1, keepdim=True)
    cyl = torch.tensor([[[5.0, 0.0, 1.0, 3.0]]])  # centre (5,0), r=1, h=3
    t = cast(o, d, cyl, None, ground=True)
    assert t[0, 0].item() == pytest.approx(2.0, abs=1e-5)  # straight down
    assert t[0, 1].item() == pytest.approx(4.0, abs=1e-5)  # cylinder side
    assert t[0, 2].item() == pytest.approx(math.sqrt(2.0), abs=1e-5)  # ground before cylinder
    # from (0,0,5) slope -0.6 -> reaches z=3 (top cap) at x=3.33 (outside), then side? top cap region x in [4,6]
    # z(x) = 5 - 0.6 x / 1 ... direction normalized; hits top cap z=3 at x=3.333 (miss, r<|x-5|), side at x=4 z=2.6
    assert t[0, 3].item() == pytest.approx(math.hypot(4.0, 2.4), abs=1e-4)
    assert t[0, 4].item() == pytest.approx(0.0, abs=1e-6)  # origin inside cylinder
    box = torch.tensor([[[3.0, 0.0, 1.0, 0.5, 0.5, 0.5, math.pi / 4]]])  # rotated cube
    tb = cast(o[:, 1:2], d[:, 1:2], None, box, ground=False)
    assert tb[0, 0].item() == pytest.approx(3.0 - 0.5 * math.sqrt(2.0), abs=1e-5)


def test_pinhole_symmetry():
    d = pinhole_ray_dirs(64, 48, 87.0, "cpu").view(48, 64, 3)
    az_left = math.degrees(math.atan2(d[24, 0, 1], d[24, 0, 0]))
    az_right = math.degrees(math.atan2(d[24, -1, 1], d[24, -1, 0]))
    assert az_left == pytest.approx(-az_right, abs=1e-4)
    # outermost pixel centres are half a pixel inside the FOV edge
    fx = 32 / math.tan(math.radians(43.5))
    assert az_left == pytest.approx(math.degrees(math.atan(31.5 / fx)), abs=1e-3)


def test_depth_noise_statistics():
    torch.manual_seed(4)
    cfg = DepthNoiseCfg(stereo_coeff=0.005, base_std=0.005, dropout_prob=0.02, quantization=0.0)
    d = torch.full((200, 48, 64), 4.0)
    n = apply_depth_noise(d, cfg)
    valid = n > 0
    assert (~valid).float().mean().item() == pytest.approx(0.02, abs=0.003)
    assert n[valid].std().item() == pytest.approx(0.005 + 0.005 * 16, rel=0.05)
    far = apply_depth_noise(torch.full((1, 4, 4), 50.0), DepthNoiseCfg(enabled=False))
    assert far.max().item() == cfg.max_range
    # regression: no-return pixels (inf / huge) must read max range (except dropouts), never random values
    sky = apply_depth_noise(torch.full((50, 48, 64), float("inf")), DepthNoiseCfg(dropout_prob=0.0))
    assert torch.all(sky == DepthNoiseCfg().max_range)
    near = apply_depth_noise(torch.full((10, 8, 8), 0.1), cfg)
    assert torch.all(near == cfg.invalid_value)


def test_wind_turbulence_std():
    torch.manual_seed(5)
    w = WindModel(WindCfg(enabled=True, mean_speed_range=(0.0, 0.0), vertical_std=0.0, turbulence_std=0.8,
                          turbulence_tau=1.0, gust_prob_per_s=0.0), 20000, "cpu")
    w.reset(torch.arange(20000))
    for _ in range(100):
        v = w.step(0.02)
    assert v.std(0).mean().item() == pytest.approx(0.8, rel=0.05)
