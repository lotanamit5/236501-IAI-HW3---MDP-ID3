from copy import deepcopy
import numpy as np

def get_legal_states(mdp):
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if board[i][j] != "WALL"
    
def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    delta = 0
    gamma = mdp.gamma
    U_ = U_init
    
    while delta < epsilon * (1 - gamma) / gamma:
        U = U_
        delta = 0
        
        for i, j in [(i,j) for i in  for j in range(mdp.num_col)]:
            U_[i][j] = mdp.board[]
        
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
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
