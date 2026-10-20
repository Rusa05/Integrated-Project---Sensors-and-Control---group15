"""Named test scenarios: nominal plus disturbed / degraded variants used for Part E.

Each returns (WorldSpec, Disturbances) for a given seed, so every trial is reproducible.
"""
from __future__ import annotations

import numpy as np

from . import world as W
from .sim import Disturbances


def nominal(seed):
    return W.nominal_world(), Disturbances()


def random_layout(seed):
    """Obstacles moved up to ±0.25 m and rotated, start pose ±0.2 m / ±0.6 rad."""
    return W.randomised_world(np.random.default_rng(seed)), Disturbances()


def push(seed):
    """A 0.3 s lateral shove of 25 N mid-mission (an impulse of 7.5 N s on a ~1.9 kg robot)."""
    rng = np.random.default_rng(seed)
    spec = W.randomised_world(rng)
    ang = rng.uniform(0, 2 * np.pi)
    return spec, Disturbances(push_time=rng.uniform(20, 35),
                              push_force=(25 * np.cos(ang), 25 * np.sin(ang)))


def slip(seed):
    """A 1.2 x 1.6 m low-friction patch (mu = 0.15) across the route."""
    rng = np.random.default_rng(seed)
    spec = W.randomised_world(rng)
    spec.slip_patches = [(3.0 + rng.uniform(-0.3, 0.3), 1.0 + rng.uniform(-0.2, 0.2), 0.6, 0.8, 0.15)]
    return spec, Disturbances()


def encoder_bias(seed):
    """Uncalibrated wheel-radius mismatch: left encoder reads 4 % high, right 1 % low."""
    rng = np.random.default_rng(seed)
    return W.randomised_world(rng), Disturbances(enc_scale=(1.04, 0.99))


def camera_degraded(seed):
    """Pixel noise x3 and 40 % of camera frames dropped."""
    rng = np.random.default_rng(seed)
    return W.randomised_world(rng), Disturbances(pixel_noise_scale=3.0, camera_dropout=0.4)


def combined(seed):
    """Everything at once: random layout, push, encoder bias, degraded camera, noisy LiDAR."""
    rng = np.random.default_rng(seed)
    spec = W.randomised_world(rng)
    ang = rng.uniform(0, 2 * np.pi)
    return spec, Disturbances(enc_scale=(1.03, 0.99), push_time=rng.uniform(20, 35),
                              push_force=(20 * np.cos(ang), 20 * np.sin(ang)),
                              pixel_noise_scale=2.0, camera_dropout=0.25, lidar_noise_scale=2.0)


SCENARIOS = {f.__name__: f for f in (nominal, random_layout, push, slip, encoder_bias,
                                     camera_degraded, combined)}
