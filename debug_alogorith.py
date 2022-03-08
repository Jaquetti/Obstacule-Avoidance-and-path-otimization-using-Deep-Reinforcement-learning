
#!/usr/bin/python3
import gym

env = gym.make('Pendulum-v1')
env.reset()
for _ in range(10):
    env.render()
    env.step(env.action_space.sample()) # take a random action
    print(env.action_space.sample())
env.close()