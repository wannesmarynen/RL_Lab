"""
Module containing the agent classes to solve a Bandit problem.

Complete the code wherever TODO is written.
Do not forget the documentation for every class and method!
An example can be seen on the Bandit_Agent and Random_Agent classes.
"""
# -*- coding: utf-8 -*-
import numpy as np
from utils import softmax, my_random_choice


class Bandit_Agent(object):
    """
    Abstract Agent to solve a Bandit problem.

    Contains the methods learn() and act() for the base life cycle of an agent.
    The reset() method reinitializes the agent.
    The minimum requirment to instantiate a child class of Bandit_Agent
    is that it implements the act() method (see Random_Agent).
    """

    def __init__(self, k: int, **kwargs):
        """
        Simply stores the number of arms of the Bandit problem.
        The __init__() method handles hyperparameters.
        Parameters
        ----------
        k: positive int
            Number of arms of the Bandit problem.
        kwargs: dictionary
            Additional parameters, ignored.
        """
        self.k = k

    def reset(self):
        """
        Reinitializes the agent to 0 knowledge, good as new.

        No inputs or outputs.
        The reset() method handles variables.
        """
        pass

    def learn(self, a: int, r: float):
        """
        Learning method. The agent learns that action a yielded reward r.
        Parameters
        ----------
        a: positive int < k
            Action that yielded the received reward r.
        r: float
            Reward for having performed action a.
        """
        pass

    def act(self) -> int:
        """
        Agent's method to select a lever (or Bandit) to pull.
        Returns
        -------
        a : positive int < k
            The action the agent chose to perform.
        """
        raise NotImplementedError("Calling method act() in Abstract class Bandit_Agent")


class Random_Agent(Bandit_Agent):
    """
    This agent doesn't learn, just acts purely randomly.
    Good baseline to compare to other agents.
    """

    def act(self):
        """
        Random action selection.
        Returns
        -------
        a : positive int < k
            A randomly selected action.
        """
        return np.random.randint(self.k)


class EpsGreedy_SampleAverage(Bandit_Agent):
    # TODO: implement this class following the formalism above.
    # This class uses Sample Averages to estimate q; others are non stationary.
    def __init__(self, eps: float, **kwargs):
        super(EpsGreedy_SampleAverage, self).__init__(**kwargs)
        self.eps = eps
        self.q = [0] * self.k
        self.steps = [0] * self.k

    def act(self) -> int:
        if np.random.random() > self.eps:
            action = max(enumerate(self.q), key=lambda q: q[1])[0]
        else:
            action = np.random.randint(self.k)
        self.steps[action] += 1
        return action

    def learn(self, a: int, r: float):
        self.q[a] += (1 / self.steps[a]) * (r - self.q[a])

    def reset(self):
        self.q = [0] * self.k
        self.steps = [0] * self.k


class EpsGreedy(EpsGreedy_SampleAverage):
    # TODO: implement this class following the formalism above.
    # Non stationary agent with q estimating and eps-greedy action selection.
    def __init__(self, alpha, **kwargs):
        super(EpsGreedy, self).__init__(**kwargs)
        self.alpha = alpha

    def learn(self, a: int, r: float):
        self.q[a] += self.alpha * (r - self.q[a])


class OptimisticGreedy(EpsGreedy):
    # TODO: implement this class following the formalism above.
    # Same as above but with optimistic starting values.
    def __init__(self, q0: int, **kwargs):
        super(OptimisticGreedy, self).__init__(**kwargs)
        self.q0 = q0
        self.q = [q0] * self.k

    def act(self) -> int:
        return max(enumerate(self.q), key=lambda q: q[1])[0]

    def reset(self):
        self.q = [self.q0] * self.k


class UCB(Bandit_Agent):
    # TODO: implement this class following the formalism above.
    def __init__(self, alpha, c, **kwargs):
        super(UCB, self).__init__(**kwargs)
        self.total_steps = 0
        self.steps = [0] * self.k
        self.alpha = alpha
        self.q = [0] * self.k
        self.c = c

    def act(self):
        if self.total_steps < self.k:
            action = self.total_steps
            self.total_steps += 1
            self.steps[action] += 1
        else:
            temp = []
            for i in range(self.k):
                temp.append(self.q[i] + self.c * (np.sqrt(np.log(self.total_steps) / (self.steps[i]))))
            action = max(enumerate(temp), key=lambda q: q[1])[0]
            self.total_steps += 1
            self.steps[action] += 1
        return action

    def learn(self, a: int, r: float):
        self.q[a] += (1 / self.steps[a]) * (r - self.q[a])

    def reset(self):
        self.total_steps = 0
        self.steps = [0] * self.k


class Gradient_Bandit(Bandit_Agent):
    # TODO: implement this class following the formalism above.
    # If you want this to run fast, use the my_random_choice function from
    # utils instead of np.random.choice to sample from the softmax
    # You can also find the softmax function in utils.
    def __init__(self, alpha, **kwargs):
        super(Gradient_Bandit, self).__init__(**kwargs)
        self.pi = [0] * self.k
        self.pref = [0] * self.k
        self.alpha = alpha
        self.steps = 0
        self.avarageReward = 0

    def learn(self, a: int, r: float):
        self.steps += 1
        self.avarageReward += ((1 / self.steps) * (r - self.avarageReward))

        self.pref[a] += self.alpha * (r - self.avarageReward) * (1 - self.pi[a])
        for i in range(self.k):
            if i != a:
                self.pref[a] -= self.alpha * (r - self.avarageReward) * self.pi[a]

    def act(self) -> int:
        return max(enumerate(self.pref), key=lambda q: q[1])[0]

    def reset(self):
        self.pi = [0] * self.k
        self.pref = [0] * self.k
