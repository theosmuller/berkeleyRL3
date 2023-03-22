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

import cmath
import mdp, util
from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
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
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Write value iteration code here

        # a cada iteração
        for i in range(0, iterations):
            newStateValues = util.Counter()
            # para cada estado do processo de decisão de Markov
            for s in self.mdp.getStates():
                # inicializa uma variável para armazenar os Qstates partindo do estado atual
                Qs = util.Counter()
                # computa os resultados de cada ação para o estado atual e armazena na variável Qs
                for a in self.mdp.getPossibleActions(s):
                    Qs[a] = self.computeQValueFromValues(s, a)
                # valor do estado atual = melhor valor dos Qstates
                newStateValues[s] = Qs[Qs.argMax()]

            self.values = newStateValues


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
        # começa com valor zerado
        actionValue = 0.0
        # para cada par de proximo estado e probabilidade do estado ocorrer dado o atual e ação escolhida
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            # agrega ao valor total da ação a probabilidade da recompensa ocorrer + o desconto da transição de estado
            actionValue += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.getValue(nextState))
        return actionValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        bestAction = None # retornando none se não houver ações legais
        bestVal = -cmath.inf # qualquer valor é melhor que o inicial

        # para cada ação possível partindo do estado atual
        for a in self.mdp.getPossibleActions(state):
            # aplica o Qlearning para o par estado ação e armazena o valor
            val = self.computeQValueFromValues(state, a)
            # se o valor for melhor que o atual melhor, atualiza
            if(val > bestVal):
                bestVal = val
                bestAction = a

        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
