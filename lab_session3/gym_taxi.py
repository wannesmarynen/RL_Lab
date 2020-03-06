"""
This is a example code of gym and test your virtual env.
Action Space Discrete(6)
State Space Discrete(500)
The filled square represents the taxi, which is yellow without a passenger and green with a passenger.
The pipe ("|") represents a wall which the taxi cannot cross.
R, G, Y, B are the possible pickup and destination locations. The blue letter represents the current passenger pick-up location, and the purple letter is the current destination

Actions
-------
0 = south
1 = north
2 = east
3 = west
4 = pickup
5 = dropoff
"""

import gym
from IPython.display import clear_output
from time import sleep
<<<<<<< HEAD

env = gym.make("Taxi-v3")
=======
env = gym.make("Taxi-v2")
>>>>>>> upstream/master
env.render()
env.reset()

"""
We'll create an infinite loop which runs until one passenger reaches 
one destination (one episode), or in other words, when the received reward is 20.
The env.action_space.sample() method automatically selects one random action 
from set of all possible actions.
"""

env.s = 328  # set environment to illustration's state

epochs = 0
penalties, reward = 0, 0

<<<<<<< HEAD
frames = []  # for animation
=======
frames = [] # for animation
>>>>>>> upstream/master

done = False

while not done:
<<<<<<< HEAD
    action = env.action_space.sample()  # returns a random action(0-5)
    new_state, reward, done, info = env.step(action)  # take the action and get reward and next state

    if reward == -10:
        penalties += 1

=======
    action = env.action_space.sample() # returns a random action(0-5)
    new_state, reward, done, info = env.step(action) # take the action and get reward and next state

    if reward == -10:
        penalties += 1
    
>>>>>>> upstream/master
    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': new_state,
        'action': action,
        'reward': reward
<<<<<<< HEAD
    }
=======
		}
>>>>>>> upstream/master
    )

    epochs += 1

<<<<<<< HEAD
=======

>>>>>>> upstream/master
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
<<<<<<< HEAD
        print(frame['frame'])
=======
        print(frame['frame'].getvalue())
>>>>>>> upstream/master
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)

<<<<<<< HEAD

=======
>>>>>>> upstream/master
print_frames(frames)
