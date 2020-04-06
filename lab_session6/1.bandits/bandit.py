"""
Module containing the k-armed bandit problem
Complete the code wherever TODO is written.
Do not forget the documentation for every class and method!
We expect all classes to follow the Bandit abstract object formalism.
"""
# -*- coding: utf-8 -*-
import numpy as np


class Bandit(object):
    """
    Abstract concept of a Bandit, i.e. Slot Machine, the Agent can pull.

    A Bandit is a distribution over reals.
    The pull() method samples from the distribution to give out a reward.
    """

    def __init__(self, **kwargs):
        """
        Empty for our simple one-armed bandits, without hyperparameters.
        Parameters
        ----------
        **kwargs: dictionary
            Ignored additional inputs.
        """
        pass

    def reset(self):
        """
        Reinitializes the distribution.
        """
        pass

    def pull(self) -> float:
        """
        Returns a sample from the distribution.
        """
        raise NotImplementedError("Calling method pull() in Abstract class Bandit")


class Mixture_Bandit_NonStat:
    """ A Mixture_Bandit_NonStat is a 2-component Gaussian Mixture
    reward distribution (sum of two Gaussians with weights w and 1-w in [O,1]).

    The two means are selected according to N(0,1) as before.
    The two weights of the gaussian mixture are selected uniformely.
    The Gaussian mixture in non-stationary: the means AND WEIGHTS move every
    time-step by an increment epsilon~N(m=0,std=0.01)"""

    # TODO: Implement this class inheriting the Bandit above.
    def __init__(self, **kwargs):
        super(Mixture_Bandit_NonStat, self).__init__()
        self.mean1 = 0
        self.mean2 = 0
        self.weight = 0.5
        self.reset()

    def reset(self):
        self.mean1 = np.random.normal(scale=1, loc=0)
        self.mean2 = np.random.normal(scale=1, loc=0)

    def pull(self, index=0) -> float:
        result = self.weight * np.random.normal(scale=1, loc=self.mean1) + (1 - self.weight) * np.random.normal(scale=1,
                                                                                                                loc=self.mean1)
        self.mean1 += np.random.normal(scale=0.01, loc=0)
        self.mean2 += np.random.normal(scale=0.01, loc=0)
        self.weight = np.random.uniform()
        return result

    def best_option(self, index) -> bool:
        return True


class Gaussian_Bandit_NonStat(Bandit):
    # TODO: implement this class following the formalism above.
    # Reminder: the distribution mean changes each step over time,
    # with increments following N(m=0,std=0.01)
    def __init__(self, **kwargs):
        super(Gaussian_Bandit_NonStat, self).__init__()
        self.mean = 0
        self.reset()

    def reset(self):
        self.mean = np.random.normal(scale=1, loc=0)

    def pull(self, index=0) -> float:
        result = np.random.normal(scale=1, loc=self.mean)
        self.mean += np.random.normal(scale=0.01, loc=0)
        return result

    def best_option(self, index) -> bool:
        return True


class KBandit_NonStat:
    """ Set of K Mixture_Bandit_NonStat Bandits.
    The Bandits are non stationary, i.e. every pull changes all the
    distributions.

    This k-armed Bandit has:
    * an __init__ method to initialize k
    * a reset() method to reset all Bandits
    * a pull(lever) method to pull one of the Bandits; + non stationarity
    """

    def __init__(self, k: int, **kwargs):
        super(KBandit_NonStat, self).__init__()
        self.Bandits = [Gaussian_Bandit_NonStat() for _ in range(k)]

    def reset(self):
        for b in self.Bandits:
            b.reset()

    def pull(self, index: int = 0) -> float:
        for i, b in enumerate(self.Bandits):
            temp = b.pull()
            if i == index:
                final = temp
        return final

    def best_option(self, i) -> bool:
        index_best = 0
        mean_best = float("-inf")
        for index, b in enumerate(self.Bandits):
            b = self.Bandits[index]
            if (b.mean > mean_best):
                mean_best = b.mean
                index_best = index
        return i == index_best
