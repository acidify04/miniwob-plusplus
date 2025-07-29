import time
import gymnasium
from miniwob.action import ActionTypes
from miniwob.fields import field_lookup
import numpy as np
import random

# Import `custom_registry.py` above to register the task.
import custom_registry
gymnasium.register_envs(custom_registry)

# Create an environment.
env = gymnasium.make('miniwob/custom-v0', render_mode='human')

try:
    for i in range(10):
        # Start a new episode.
        observation, info = env.reset()
        #print(observation["dom_buttons"])

        for button in observation["dom_elements"]:
            if button["text"] == "Click Me!":
                break

        #print(button)
        click_x = button["left"].item() + button["width"].item() / 2
        click_y = button["top"].item() + button["height"].item() / 2
        
        print(click_x, click_y)

        if i == 0:
            action = env.unwrapped.create_action(
                ActionTypes.MOVE_COORDS, coords=np.array([0., 0.], dtype=np.float64)
                )
            observation, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.5)

        action = env.unwrapped.create_action(
            ActionTypes.MOVE_COORDS, coords=np.array([click_x, click_y], dtype=np.float64)
            )
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.5)

        action = env.unwrapped.create_action(ActionTypes.CLICK_COORDS, coords=np.array([click_x, click_y], dtype=np.float64))
        observation, reward, terminated, truncated, info = env.step(action)
        #print(reward)
        env.render()
        click_reward += reward
    
    # Check if the action was correct. 
        assert reward >= 0
        # assert terminated is True
        time.sleep(0.5)

finally:
  env.close()