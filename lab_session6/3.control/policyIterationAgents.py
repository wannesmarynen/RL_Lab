# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import numpy as np
from learningAgents import ValueEstimationAgent


class PolicyIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PolicyIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs policy iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        print("using discount {}".format(discount))
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.policy = {}



        delta = 0.01
        # TODO: Implement Policy Iteration.
        # Exit either when the number of iterations is reached,
        # OR until convergence (L2 distance < delta).
        # Print the number of iterations to convergence.
        # To make the comparison FAIR, one iteration is a single sweep over states.
        # Compute the number of steps until policy convergence, but do not stop
        # the algorithm until values converge.

        for s in self.mdp.getStates():
            a = self.mdp.getPossibleActions(s)
            if len(a) > 0:
                self.policy[s] = np.random.choice(a)
            else:
                self.policy[s] = -1

        final = False

        for i in range(self.iterations):
            changed = False
            for s in self.mdp.getStates():
                a = self.policy[s]
                if a != -1:
                    self.values[s] = np.sum([prob * (self.mdp.getReward(s, a, sa) + self.discount * self.values[sa]) for sa, prob in self.mdp.getTransitionStatesAndProbs(s, a)])

            for s in self.mdp.getStates():
                current_best = self.values[s]
                for a in self.mdp.getPossibleActions(s):
                    q_sa = np.sum([prob * (self.mdp.getReward(s, a, sa) + self.discount * self.values[sa]) for sa, prob in self.mdp.getTransitionStatesAndProbs(s, a)])
                    if q_sa > current_best:
                        self.policy[s] = a
                        current_best = q_sa
                        changed = True

            if not changed and not final:
                print('itterations: ', i)
                final = True

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # TODO: Implement this function according to the doc
        newstates = self.mdp.getTransitionStatesAndProbs(state, action)
        newstate = (-1, -1)
        max = -float('Inf')
        for sa, p in newstates:
            if p > newstate[1]:
                newstate = (sa, p)
        if max < self.values[newstate[0]]:
            max = self.values[newstate[0]]
        return max

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # TODO: Implement according to the doc
        max = -float('Inf')
        action = -1
        if self.mdp.isTerminal(state):
            return None
        for a in self.mdp.getPossibleActions(state):
            newstates = self.mdp.getTransitionStatesAndProbs(state, a)
            newstate = (-1, -1)
            for sa, p in newstates:
                if p > newstate[1]:
                    newstate = (sa, p)
            if max < self.values[newstate[0]]:
                max = self.values[newstate[0]]
                action = a
        return action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
