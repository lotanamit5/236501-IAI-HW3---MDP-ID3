from copy import deepcopy
import numpy as np
from mdp import MDP

actions_to_idx = {"UP":0, "DOWN":1, "RIGHT":2, "LEFT":3}
idx_to_actions = ["UP", "DOWN", "RIGHT", "LEFT"]

@staticmethod
def deepcopy_with_walls(mdp, U):
    U_ = deepcopy(U)
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL":
                U_[i][j] = None
    return U_

@staticmethod
def deepcopy_with_final_states(mdp, U):
    U_ = deepcopy(U)
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if (i, j) in mdp.terminal_states:
                U_[i][j] = None
    return U_

@staticmethod
def get_states(mdp: MDP):
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL":
                continue
            yield (i, j)

@staticmethod
def reward(mdp: MDP, state):
    i, j = state
    return float(mdp.board[i][j])

@staticmethod
def action_from_policy(policy, s_from):
    i, j = s_from
    return policy[i][j]

@staticmethod
def transition(mdp, s_from, s_to, wanted_action):
    if s_from in mdp.terminal_states:
        return 0
    
    possible_actions = [action for action in actions_to_idx if mdp.step(s_from, action) == s_to]
    
    prob = 0
    for action in possible_actions:
        prob += mdp.transition_function[wanted_action][actions_to_idx[action]]
    return prob

def total_utility(mdp, U, state, action):
    utilities = [transition(mdp, state, s_to, action)*get_utility(U, s_to) for s_to in get_states(mdp)]
    return sum(utilities)

def get_utility(U, state):
    i, j = state
    return U[i][j]

def utility(mdp, U, state, action):
    actions = ["UP", "DOWN", "RIGHT", "LEFT"]
    profits = [U[i_][j_] for i_, j_ in [mdp.step(state, a) for a in actions]]
    probs = mdp.transition_function[action]
    muls = [profit * prob for profit, prob in zip(profits, probs)]
    return sum(muls)

def get_legal_states(mdp):
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL" or (i, j) in mdp.terminal_states:
                continue
            yield (i, j)
    
def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    delta = float('inf')
    gamma = mdp.gamma
    U_ = deepcopy_with_walls(mdp, U_init)
    U = None
    
    for i, j in mdp.terminal_states:
        U_[i][j] = float(mdp.board[i][j])
    
    while delta > epsilon * (1 - gamma) / gamma:
        delta = 0
        U = deepcopy(U_)
        
        for i, j in get_legal_states(mdp):
            reward = float(mdp.board[i][j])
            sums = [utility(mdp, U, (i,j), action) for action in mdp.actions.keys()]
            U_[i][j] = reward + gamma * max(sums)
            
            if abs(U_[i][j] - U[i][j]) > delta:
                delta = abs(U_[i][j] - U[i][j])
            
    return U

    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    policy = deepcopy_with_final_states(mdp, U)
    
    for i, j in get_legal_states(mdp):
        sums = [utility(mdp, U, (i,j), action) for action in mdp.actions.keys()]
        best_index = sums.index(max(sums))
        policy[i][j] = ["UP", "DOWN", "RIGHT", "LEFT"][best_index]
    
    return policy
    # ========================


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    
    rewards = np.array([reward(mdp, s) for s in get_states(mdp)])
    
    transitions = np.array([[transition(mdp, s_from, s_to, action_from_policy(policy, s_from)) 
                             for s_to in get_states(mdp)] for s_from in get_states(mdp)])
    
    utility = np.linalg.inv(np.eye(len(rewards)) - mdp.gamma * transitions) @ rewards
    
    # Turn utility vector to a map
    U = deepcopy(policy)
    for s, u in zip(get_states(mdp), utility.tolist()):
        i, j = s
        U[i][j] = u
    
    return U
    
    # ========================

    
def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    
    policy = deepcopy(policy_init)
    
    changed = True
    while changed:
        U = policy_evaluation(mdp, policy)
        changed = False
        
        for s in get_states(mdp):
            utilities = [total_utility(mdp, U, s, a) for a in actions_to_idx]
            max_util = max(utilities)
            total_util = total_utility(mdp, U, s, action_from_policy(policy, s))
            if max_util > total_util and not np.isclose(max_util, total_util, rtol=1e-6):
                i, j = s
                policy[i][j] = idx_to_actions[utilities.index(max(utilities))]
                changed = True
    return policy

    # ========================
