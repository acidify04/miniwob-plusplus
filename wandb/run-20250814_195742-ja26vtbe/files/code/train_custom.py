import gymnasium as gym
import numpy as np
from gymnasium import spaces
from miniwob.action import ActionTypes
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
import wandb
from wandb.integration.sb3 import WandbCallback

# ----------------------------
# Custom Environment
# ----------------------------
class MiniWoBClickImageEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = gym.make('miniwob/custom-v0')

        # screenshot 크기
        sample_obs, _ = self.env.reset()
        h, w, c = sample_obs['screenshot'].shape
        self.height, self.width = h, w

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, c), dtype=np.uint8
        )
        # 클릭 좌표 (정규화)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs['screenshot'], info

    def step(self, action):
        coords = np.array([action[0] * self.width, action[1] * self.height], dtype=np.float32)

        # Move
        move_action = self.env.unwrapped.create_action(ActionTypes.MOVE_COORDS, coords=coords)
        self.env.step(move_action)

        # Click
        click_action = self.env.unwrapped.create_action(ActionTypes.CLICK_COORDS, coords=coords)
        obs, reward, terminated, truncated, info = self.env.step(click_action)

        # ----------------------------
        # Reward shaping: 클릭 위치와 target 간 거리 기반 보너스
        # ----------------------------
        if 'goal' in info:  # 환경이 목표 좌표를 info로 준다고 가정
            goal_coords = np.array(info['goal'])
            dist = np.linalg.norm(coords - goal_coords)
            max_dist = np.linalg.norm([self.width, self.height])
            shaped_reward = max(0, 1 - dist / max_dist)
            reward += shaped_reward * 0.5  # reward shaping 비중
        return obs['screenshot'], reward, terminated, truncated, info

    def close(self):
        self.env.close()

# ----------------------------
# W&B 초기화
# ----------------------------
wandb.init(
    project="miniwob-click",
    config={
        "policy_type": "CnnPolicy",
        "total_timesteps": 100000,
        "n_envs": 8,
        "learning_rate": 5e-5,
        "ent_coef": 0.01
    },
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True
)

# ----------------------------
# Vectorized 환경
# ----------------------------
env = make_vec_env(MiniWoBClickImageEnv, n_envs=8)
env = VecTransposeImage(env)

# ----------------------------
# PPO 모델
# ----------------------------
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=5e-5,
    n_steps=256,         # 더 자주 업데이트
    ent_coef=0.01,       # exploration 강화
    batch_size=64,
    gamma=0.95,
    tensorboard_log="./ppo_wob_tensorboard/"
)

# ----------------------------
# 학습
# ----------------------------
model.learn(
    total_timesteps=100000,
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{wandb.run.id}",
        verbose=2
    )
)
model.save("click_model_cnn")
