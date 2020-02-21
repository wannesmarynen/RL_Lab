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

    def pull(self, index: int = 0) -> float:
        """
        Returns a sample from the distribution.
        """
        raise NotImplementedError("Calling method pull() in Abstract class Bandit")

    def best_option(self, index) -> bool:
        pass

class Gaussian_Bandit(Bandit):
    # TODO: implement this class following the formalism above.
    # Reminder: the Gaussian_Bandit's distribution is a fixed Gaussian.
    def __init__(self,  **kwargs):
        super(Gaussian_Bandit, self).__init__()
        self.mean = 0
        self.reset()

    def reset(self):
        self.mean = np.random.normal(scale=1, loc=0)

    def pull(self, index=0) -> float:
        return np.random.normal(scale=1, loc=self.mean)

    def best_option(self, index) -> bool:
        return True


class Gaussian_Bandit_NonStat(Bandit):
    # TODO: implement this class following the formalism above.
    # Reminder: the distribution mean changes each step over time,
    # with increments following N(m=0,std=0.01)
    def __init__(self,  **kwargs):
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


class KBandit(Bandit):
    # TODO: implement this class following the formalism above.
    # Reminder: The k-armed Bandit is a set of k Bandits.
    # In this case we mean for it to be a set of Gaussian_Bandits.
    def __init__(self, k: int,  **kwargs):
        super(KBandit, self).__init__()
        self.Bandits = [Gaussian_Bandit() for _ in range(k)]

    def reset(self):
        for b in self.Bandits:
            b.reset()

    def pull(self, index: int = 0) -> float:
        for i, b in enumerate(self.Bandits):
            temp = b.pull()
            if (i == index):
                final = temp
        return final

    def best_option(self, i) -> bool:
        index_best = 0
        mean_best = self.Bandits[0].mean
        for index, b in enumerate(self.Bandits):
            if b.mean > mean_best:
                mean_best = b.mean
                index_best = index
        return int(i) == index_best


class KBandit_NonStat(Bandit):
    # TODO: implement this class following the formalism above.
    # Reminder: Same as KBandit, with non stationary Bandits.
    def __init__(self, k: int,  **kwargs):
        super(KBandit_NonStat, self).__init__()
        self.Bandits = [Gaussian_Bandit_NonStat() for _ in range(k)]

    def reset(self):
        for b in self.Bandits:
            b.reset()

    def pull(self, index: int = 0) -> float:
        for i, b in enumerate(self.Bandits):
            temp = b.pull()
            if(i == index):
                final = temp
        return final

    def best_option(self, i) -> bool:
        index_best = 0
        mean_best = float("-inf")
        for index, b in enumerate(self.Bandits):
            b = self.Bandits[index]
            if(b.mean > mean_best):
                mean_best = b.mean
                index_best = index
        return i == index_best