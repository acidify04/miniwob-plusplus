import gymnasium as gym
import numpy as np
from gymnasium import spaces
from miniwob.action import ActionTypes
import custom_registry
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

class MiniWoBClickImageEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = gym.make('miniwob/custom-v0')

        # screenshot 크기
        sample_obs, _ = self.env.reset()
        h, w, c = sample_obs['screenshot'].shape

        self.height = h
        self.width = w

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(h, w, c),
            dtype=np.uint8
        )

        # 클릭 좌표 (정규화된 x, y)
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(2,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs['screenshot'], info

    def step(self, action):
        coords = np.array([action[0] * self.width, action[1] * self.height], dtype=np.float32)

        move_action = self.env.unwrapped.create_action(
            ActionTypes.MOVE_COORDS, coords=coords
        )
        self.env.step(move_action)

        click_action = self.env.unwrapped.create_action(
            ActionTypes.CLICK_COORDS, coords=coords
        )
        obs, reward, terminated, truncated, info = self.env.step(click_action)

        return obs['screenshot'], reward, terminated, truncated, info

    def close(self):
        self.env.close()

wandb.init(
    project="miniwob-click",   # W&B 프로젝트 이름
    config={                  # 하이퍼파라미터 기록
        "policy_type": "CnnPolicy",
        "total_timesteps": 10000,
        "env_name": "MiniWoBClickImageEnv",
        "n_envs": 8,
        "learning_rate": 5e-5,
        "ent_coef": 0.01
    },
    sync_tensorboard=True,     # tensorboard 로그도 같이 동기화
    monitor_gym=True,          # gym 환경 모니터링
    save_code=True             # 코드도 자동 저장
)

# 벡터화된 환경 생성
env = make_vec_env(MiniWoBClickImageEnv, n_envs=1)

# (H, W, C) → (C, H, W) 변환
env = VecTransposeImage(env)

# CNN Policy로 학습
model = PPO("CnnPolicy", env, verbose=2, n_steps=256, tensorboard_log="./ppo_wob_tensorboard/")
model.learn(
    total_timesteps=100000,
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{wandb.run.id}", # 학습된 모델도 W&B에 업로드
        verbose=2
    )
)
model.save("click_model_cnn")

env = make_vec_env(MiniWoBClickImageEnv, n_envs=256)
env = VecTransposeImage(env)

model = PPO.load("click_model_cnn", env=env)

obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
