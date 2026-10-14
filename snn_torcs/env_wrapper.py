"""
Discrete-action wrapper over TorcsEnv.

Maps discrete steering actions to continuous TORCS commands with fixed throttle.
Flattens the namedtuple observation into a stable 68-dim float32 vector.
"""
import sys
import os
import numpy as np

# repo root on sys.path so gym_torcs is importable from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gym_torcs import TorcsEnv

# Default 7-action steering space (current training config).
STEER_VALUES = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0], dtype=np.float32)
NUM_ACTIONS = 7

# Known action mappings by output_dim — used for backward-compatible eval.
# output_dim=5 was the original training config before expanding to 7.
STEER_VALUES_BY_DIM = {
    7: np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0], dtype=np.float32),
    5: np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),
}


class TorcsDiscreteWrapper:
    """Gym-like interface with discrete steering and fixed throttle."""

    def __init__(self, vision=False, throttle=True, fixed_accel=0.3,
                 max_episode_steps=1000, run_mode=None, steer_values=None,
                 center_coef=0.1, angle_coef=0.05):
        self.env = TorcsEnv(vision=vision, throttle=throttle, run_mode=run_mode)
        self.fixed_accel = fixed_accel
        self.max_steps = max_episode_steps
        self.step_count = 0
        self.last_raw_obs = None
        # Reward-shaping coefficients (defaults preserve prior behavior).
        # center_coef scales |trackPos|; progress ≈ 19/step so 0.1 is ~0.5% of
        # progress — raise to ~progress magnitude to actually incentivize centering.
        self.center_coef = center_coef
        self.angle_coef = angle_coef

        self.steer_values = steer_values if steer_values is not None else STEER_VALUES
        self.observation_dim = 68
        self.action_n = len(self.steer_values)

    # ------------------------------------------------------------------
    def _flatten_obs(self, obs):
        """Convert namedtuple observation → (68,) float32 vector.

        Fields: focus(5) + speedX(1) + speedY(1) + speedZ(1) +
                opponents(36) + rpm(1) + track(19) + wheelSpinVel(4) = 68
        rpm and wheelSpinVel are additionally normalised for stability.
        """
        parts = [
            np.asarray(obs.focus, dtype=np.float32).flatten(),
            np.atleast_1d(np.asarray(obs.speedX, dtype=np.float32)),
            np.atleast_1d(np.asarray(obs.speedY, dtype=np.float32)),
            np.atleast_1d(np.asarray(obs.speedZ, dtype=np.float32)),
            np.asarray(obs.opponents, dtype=np.float32).flatten(),
            np.atleast_1d(np.asarray(obs.rpm, dtype=np.float32)) / 10000.0,
            np.asarray(obs.track, dtype=np.float32).flatten(),
            np.asarray(obs.wheelSpinVel, dtype=np.float32).flatten() / 200.0,
        ]
        vec = np.concatenate(parts).astype(np.float32)
        return np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)

    # ------------------------------------------------------------------
    def reset(self, relaunch=False):
        obs = self.env.reset(relaunch=relaunch)
        self.step_count = 0
        self.last_raw_obs = self.env.client.S.d
        return self._flatten_obs(obs)

    def step(self, action):
        """action: int in [0, action_n) → discrete steering command.

        Returns (obs, reward_shaped, done, info).
        reward_shaped = reward_raw - 0.1*|trackPos| - 0.05*|angle|
        The trainer applies reward_scale internally.
        """
        steer = self.steer_values[action]
        continuous_action = np.array([steer, self.fixed_accel], dtype=np.float32)
        obs, reward_raw, done, info = self.env.step(continuous_action)
        self.step_count += 1
        self.last_raw_obs = self.env.client.S.d

        raw = self.last_raw_obs
        track_pos = float(raw.get("trackPos", 0.0))
        angle = float(raw.get("angle", 0.0))

        # Reward shaping: centering + alignment penalty
        center_penalty = self.center_coef * abs(track_pos)
        angle_penalty = self.angle_coef * abs(angle)
        reward_shaped = reward_raw - center_penalty - angle_penalty

        # attach done reason and per-step diagnostics
        info = dict(info)
        if done:
            info["done_reason"] = self._classify_done()
        else:
            info["done_reason"] = None

        info["speedX"] = float(raw.get("speedX", 0.0))
        info["trackPos"] = track_pos
        info["angle"] = angle
        info["action"] = int(action)
        info["steer_applied"] = float(steer)
        info["accel_applied"] = float(self.fixed_accel)
        info["distRaced"] = float(raw.get("distRaced", 0.0))
        info["reward_raw"] = float(reward_raw)
        info["center_penalty"] = float(center_penalty)
        info["angle_penalty"] = float(angle_penalty)
        info["reward_shaped"] = float(reward_shaped)

        return self._flatten_obs(obs), reward_shaped, done, info

    def _classify_done(self):
        raw = self.last_raw_obs
        track = np.array(raw.get("track", []))
        if len(track) > 0 and track.min() < 0:
            return "off_track"
        angle = raw.get("angle", 0.0)
        if isinstance(angle, (int, float)) and np.cos(angle) < 0:
            return "backward"
        return "slow_progress"

    def end(self):
        self.env.end()
