import time
import gymnasium
import miniwob
from miniwob.action import ActionTypes

gymnasium.register_envs(miniwob)

env = gymnasium.make('miniwob/click-test-2-v1', render_mode=None)

rewards = []

# Wrap the code in try-finally to ensure proper cleanup.
try:
    for i in range(10):
        # Start a new episode.
        obs, info = env.reset()
        assert obs["utterance"] == "Click button ONE."
        assert obs["fields"] == (("target", "ONE"),)
        time.sleep(2)       # Only here to let you look at the environment.
        
        # Find the HTML element with text "ONE".
        for element in obs["dom_elements"]:
            if element["text"] == "ONE":
                break

        # Click on the element.
        action = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=element["ref"])
        obs, reward, terminated, truncated, info = env.step(action)

        # Check if the action was correct. 
        print('=======================================')
        print(f'reward {i}: {reward}')      # Should be around 0.8 since 2 seconds has passed.
        print('=======================================')
        rewards.append(reward)
        assert terminated is True
        time.sleep(2)

finally:
    sum = 0
    for reward in rewards:
        sum += reward
    print(f'Average reward: {sum/10}')
    env.close()