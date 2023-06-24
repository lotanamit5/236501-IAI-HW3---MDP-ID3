from copy import deepcopy
import numpy as np

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
    U_ = deepcopy(U_init)
    U = None
    
    for i, j in mdp.terminal_states:
        U_[i][j] = float(mdp.board[i][j])
    
    while delta > epsilon * (1 - gamma) / gamma:
        delta = 0
        U = deepcopy(U_)
        
        for i, j in get_legal_states(mdp):
            reward = float(mdp.board[i][j])
            sums = []
            profits = [U[i_][j_] for i_, j_ in [mdp.step((i,j), a) for a in mdp.actions.keys()]]
            for action in mdp.actions.keys():
                probs = mdp.transition_function[action]
                muls = [profit * prob for profit, prob in zip(profits, probs)]
                sums.append(sum(muls))
            
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
    policy = deepcopy(U)
    
    for i, j in get_legal_states(mdp):
        sums = []
        profits = [U[i_][j_] for i_, j_ in [mdp.step((i,j), a) for a in mdp.actions.keys()]]
        for action in mdp.actions.keys():
            probs = mdp.transition_function[action]
            muls = [profit * prob for profit, prob in zip(profits, probs)]
            sums.append(sum(muls))
        best_index = sums.index(max(sums))
        policy[i][j] = ["UP, "DOWN", "RIGHT", "LEFT][best_index]
    return policy
    # ========================


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
