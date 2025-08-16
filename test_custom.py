from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from train_custom import MiniWoBClickImageEnv

env = make_vec_env(MiniWoBClickImageEnv, n_envs=1)
env = VecTransposeImage(env)

model = PPO.load("click_model_cnn", env=env)

n_test_episodes = 20
episode_rewards = []
episode_lengths = []

for ep in range(n_test_episodes):
    obs = env.reset()
    done = False
    ep_reward = 0.0
    ep_len = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)  # deterministic=True → 같은 입력에 항상 같은 행동
        obs, reward, done, info = env.step(action)
        ep_reward += reward[0]  # VecEnv이므로 reward가 배열
        ep_len += 1
        env.render()

    episode_rewards.append(ep_reward)
    episode_lengths.append(ep_len)
    print(f"Episode {ep+1}: reward={ep_reward:.3f}, length={ep_len}")

env.close()

print("=" * 40)
print(f"Average reward over {n_test_episodes} episodes: {np.mean(episode_rewards):.3f}")
print(f"Std of reward: {np.std(episode_rewards):.3f}")
print(f"Average episode length: {np.mean(episode_lengths):.1f}")
