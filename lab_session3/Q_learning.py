import numpy as np
import gym
import random

from gym.envs.registration import register

def allmax(a):
    """ Returns all occurences of the max """
    if len(a) == 0:
        return []
    all_ = [0]
    max_ = a[0]
    for i in range(1, len(a)):
        if a[i] > max_:
            all_ = [i]
            max_ = a[i]
        elif a[i] == max_:
            all_.append(i)
    return all_

def my_argmax(v):
    """ Breaks ties randomly. """
    return random.choice(allmax(v))

# code to set a gym config
# 4x4 environment
# kwargs = {'map_name': '4x4', 'is_slippery': False}
# 8x8 environment
kwargs = {'map_name': '8x8', 'is_slippery': True}
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs=kwargs,
    max_episode_steps=100,
    reward_threshold=0.8196
)

# code to set environment
env = gym.make("FrozenLakeNotSlippery-v0")

# actions
action_size = env.action_space.n
# statess
state_size = env.observation_space.n

# TODO Declare your q-table based on number of states and actions.


qtable = np.zeros((state_size, action_size))


class Agent(object):
    """
    Class declaring the agent. the qtable although handelled
    here gets passed from the Frozenlake.
    """

    def __init__(self, qtable):
        """
        Initialise the Agent.

        Paremeters
        -----------
        qtable: numpy 2d-array
        """
        self.qtable = qtable
        self.learning_rate = 0.1  # Learning rate
        self.gamma = 0.95  # Discounting rate

        # Exploration parameters
        self.epsilon = 1.0  # Exploration rate
        self.max_epsilon = 1.0  # Exploration probability at start
        self.min_epsilon = 0.1  # Minimum exploration probability
        self.decay_rate = 0.00000001  # Exponential decay rate for exploration prob

    def act(self, state):
        """
        Function where agent acts with policy eps-greedy.
        Epsilon is updated outside this method.

        Parameters
        ----------
        state: numpy int64
            current state of the environment

        Returns
        -------
        action: numpy int64


        """

        if np.random.rand() < self.epsilon:
            return np.random.randint(action_size)
        else:
            return my_argmax(self.qtable[state])

    def learn(self, state: np.int64, action: np.int64, reward: np.int64, new_state: np.int64):
        """
        Function to update the q table

        Parameters
        ----------
        state: numpy int64
            current state of the environment

        action: numpy int64
            action to update

        reward: int
            reward you get.

        new_state: numpy int64
            new state after action

        """

        self.qtable[state][action] += self.learning_rate * (
            reward + self.gamma*np.max(self.qtable[new_state]) - self.qtable[state][action]
        )

    def update_epsilon(self, episode):
        """
        Function to update the exploration.

        Parameters
        ----------
        episode: int
            episode number

        Returns
        -------
        """
        self.epsilon = self.min_epsilon + \
                       (self.max_epsilon - self.min_epsilon) * \
                       np.exp(-self.decay_rate * episode)


class Trainer(object):
    """Class to train the agent."""

    def __init__(self, qtable):
        """
        Initilisation of the class to train the agent.

        Parameters
        ----------
        qtable: numpy 2d array

        Returns
        -------

        """
        # config of your run.
        self.total_episodes = 200000  # Total episodes
        self.max_steps = 2000  # Max steps per episode

        # q-table
        self.qtable = qtable
        self.agent = Agent(self.qtable)

    def run(self):
        """
        Function to run the environment.

        Parameters
        ----------

        Returns
        -------
        """
        rewards = []

        # number of episodes you want your trainer to run
        for episode in range(self.total_episodes):
            state = env.reset()
            step = 0
            self.done = False
            self.total_rewards = 0

            # Number of steps in each episode
            for step in range(self.max_steps):
                exp_exp_tradeoff = random.uniform(0, 1)

                # take an action
                action = self.agent.act(state)

                # get feedback from environment
                new_state, reward, done, info = env.step(action)

                # update your qtable
                self.agent.learn(
                    state, action, reward, new_state)

                # move your agent to new state
                state = new_state

                # record your reward
                self.total_rewards = self.total_rewards + reward

                # check if you are dead move to next episode
                if done:
                    break

            episode += 1

            # update your exploration rate
            self.agent.update_epsilon(episode)

            # global reward
            rewards.append(self.total_rewards)

        # print your scores
        print("Score over time: " + str(sum(rewards) / self.total_episodes))

        # print the qtable
        print(self.agent.qtable)

        # printing epsilon
        print(self.agent.epsilon)

        return self.qtable


def test():
    """Function to test your agent."""
    counter = 0
    for episode in range(100):
        state = env.reset()
        print(type(state))
        step = 0
        done = False
        print("*****************************")
        print("EPISODE ", episode)
        for step in range(99):
            env.render()
            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(qtable[state, :])
            print(action)
            new_state, reward, done, info = env.step(action)
            print(reward)
            if done:
                if reward == 1:
                    print('\n \x1b[6;30;42m' + 'Success!' + '\x1b[0m')
                    counter += 1
                action = np.argmax(qtable[state, :])
                print(action)
                env.render()
                break
            state = new_state
    print("RESULT: ", counter, 100)
    env.close()


if __name__ == '__main__':
    # Declare your trainer
    trainer = Trainer(qtable)

    # train your agent
    qtable = trainer.run()

    # reset environment and test
    env.reset()
    env.render()
    if kwargs['map_name'] == '4x4':
        print(np.argmax(qtable, axis=1).reshape(4, 4))
    elif kwargs['map_name'] == '8x8':
        print(np.argmax(qtable, axis=1).reshape(8, 8))

    trainer = Trainer(qtable)

    # train your agent
    qtable = trainer.run()

    # reset environment and test
    env.reset()
    test()
