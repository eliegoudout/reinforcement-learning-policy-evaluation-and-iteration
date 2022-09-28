""" Written by Élie Goudout for personal use.
    Solver for exercise 4.7 (page 82) from Sutton & Barto's book on Reinforcement Learning.
    Can be found at: http://incompleteideas.net/book/RLbook2020.pdf
"""
from math import ceil
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from tqdm import tqdm

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# EXAMPLE 4.2 PARAMETERS
# Returned cars are not available for rent until the next day
avg_return_A = 3
avg_client_A = 3
avg_return_B = 2
avg_client_B = 4

max_cars_A = 20
max_cars_B = 20

max_car_move = 5

rent_fee = 10
move_fee = 2

gamma = .9
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# ==========================================================
# DEFINE ENVIRONMENT FOR INITIAL PROBLEM (Example 4.2)
# ==========================================================

states = [(nA, nB) for (nA, nB) in np.ndindex(max_cars_A + 1, max_cars_B + 1)]

# Storing Poisson probabilities needed for later computations
p_avg_return_A = [poisson.pmf(k, avg_return_A) for k in range(max_cars_A + 1)]
s_avg_return_A = [poisson.sf(k, avg_return_A) for k in range(max_cars_A + 1)]
p_avg_client_A = [poisson.pmf(k, avg_client_A) for k in range(max_cars_A + 1)]
s_avg_client_A = [poisson.sf(k, avg_client_A) for k in range(max_cars_A + 1)]
p_avg_return_B = [poisson.pmf(k, avg_return_B) for k in range(max_cars_B + 1)]
s_avg_return_B = [poisson.sf(k, avg_return_B) for k in range(max_cars_B + 1)]
p_avg_client_B = [poisson.pmf(k, avg_client_B) for k in range(max_cars_B + 1)]
s_avg_client_B = [poisson.sf(k, avg_client_B) for k in range(max_cars_B + 1)]

def poisson_prob(k, l, mode):
    if mode=='equal':
        if l==avg_return_A:
            return p_avg_return_A[k]
        if l==avg_client_A:
            return p_avg_client_A[k]
        if l==avg_return_B:
            return p_avg_return_B[k]
        if l==avg_client_B:
            return p_avg_client_B[k]
    if mode=='greater_than':
        if k < 0:
            return 1
        if l==avg_return_A:
            return s_avg_return_A[k]
        if l==avg_client_A:
            return s_avg_client_A[k]
        if l==avg_return_B:
            return s_avg_return_B[k]
        if l==avg_client_B:
            return s_avg_client_B[k]

def actions_set_initial_problem(state):
    """ Given a state (end of the day, cars at A and B), all possible actions (net number of cars
        to move from A to B over night).
        Can't move more than max_car_move or more than we have. Also don't move if other location
        is full.

    Args:
        state: nA, nB

    Returns:
        action set as a 1D-array
    """
    nA, nB = state
    return np.arange(-min(nB, max_car_move, max_cars_A - nA), min(nA, max_car_move, max_cars_B - nB) + 1)

def world_newState_initial_problem(NewA, NewB):
    """ <TO DO>
    """
    probs = np.zeros((max_cars_A + 1, max_cars_B + 1))
    rent_reward = 0
    for EndA, EndB in np.ndindex(max_cars_A + 1, max_cars_B + 1): # possible new states from (NewA, NewB) at beginning of day
        r = 0
        minA = max(0, NewA - EndA) # nb min of client_A
        minB = max(0, NewB - EndB) # nb min of client_B
        for c_A, c_B in np.ndindex(NewA + 1 - minA, NewB + 1 - minB): # case NewA corresponds to "NewA or more"
            client_A = c_A + minA
            client_B = c_B + minB
            r = rent_fee * (client_A + client_B)
            p = 1.
            # location A
            ## les clients 
            if client_A < NewA:
                p *= poisson_prob(client_A, avg_client_A, 'equal')
            else: # had NewA clients OR MORE
                p *= poisson_prob(client_A - 1, avg_client_A, 'greater_than')
            ## les retours
            return_A = EndA - NewA + client_A
            if EndA < max_cars_A:
                p *= poisson_prob(return_A, avg_return_A, 'equal')
            else: # had return_A OR MORE returns
                p *= poisson_prob(return_A - 1, avg_return_A, 'greater_than')
            # location B
            ## les clients
            if client_B < NewB:
                p *= poisson_prob(client_B, avg_client_B, 'equal')
            else: # had NewB clients OR MORE
                p *= poisson_prob(client_B - 1, avg_client_B, 'greater_than')
            ## les retours
            return_B = EndB - NewB + client_B
            if EndB < max_cars_B:
                p *= poisson_prob(return_B, avg_return_B, 'equal')
            else: # had return_B OR MORE returns
                p *= poisson_prob(return_B - 1, avg_return_B, 'greater_than')
            probs[EndA, EndB] += p
            rent_reward += r*p
    return probs, rent_reward

# compute entire dynamics
probs_from_new_state_initial_problem = np.zeros((max_cars_A + 1, max_cars_B + 1, max_cars_A + 1, max_cars_B + 1)) # NewA, NewB, EndA, EndB
rent_reward_from_new_state_initial_problem = np.zeros((max_cars_A + 1, max_cars_B + 1)) # NewA, NewB

def compute_dynamics_initial_problem(): # allows to access in O(1) the probability to go from one "state in the morning" (i.e. after car movments) to "state in the evening" (actual state)
    """ <TO DO>
    """
    print("Computing all the world's dynamics (relative to rentals and returns during a day not car movements)...")
    for NewA, NewB in tqdm(np.ndindex(max_cars_A + 1, max_cars_B + 1), total=(max_cars_A + 1)*(max_cars_B + 1)):
        probs_from_new_state_initial_problem[NewA, NewB], rent_reward_from_new_state_initial_problem[NewA, NewB] = world_newState_initial_problem(NewA, NewB)
    print("Done")

compute_dynamics_initial_problem()

def world_initial_problem(state, action): # returns prob for every possible next state and expected reward for this state-action
    """ <TO DO>
    """
    nA, nB = state
    newA, newB = nA - action, nB + action
    probs = probs_from_new_state_initial_problem[newA, newB]
    # reward for renting cars
    rent_reward = rent_reward_from_new_state_initial_problem[newA, newB]
    # (negative) reward for moving cars at night
    move_reward = - move_fee * abs(action)
    # total reward
    reward = rent_reward + move_reward
    return (probs, reward)

# ==========================================================
# REINFORCEMENT LEARNING INITIAL PROBLEM
# ==========================================================

# world maps (state, action) to (p(s'|s, a), expected reward).
# p is a 2D matrix. If state s'=(a,b), then p=p[a,b]
def policy_evaluation(world, policy, V_init, epsilon=0.001):
    """ <TO DO>
    """
    V = V_init
    while True:
        Delta = 0
        for state in states:
            nA, nB = state
            action = policy[nA, nB]
            probs, reward = world(state, action)
            v = V[nA, nB]
            # update rule for value function
            V[nA, nB] = reward + gamma * np.tensordot(probs, V)
            Delta = max(Delta, abs(v - V[nA, nB]))
        if Delta < epsilon:
            return V

# world maps (state, action) to (p(s'|s, a), expected reward).
# p is a 2D matrix. If state s'=(a,b), then p=p[a,b]
def policy_improvement(world, actions_set, policy, V):
    """ <TO DO>
    """
    history = [(np.copy(policy), np.copy(V))]
    while True:
        policy_stable = True
        for state in states:
            nA, nB = state
            old_policy = policy[nA, nB]
            # find best action according to V (update rule for pi)
            maxV = -np.inf
            argmaxV = 0
            for action in actions_set(state):
                probs, reward = world(state, action)
                v = reward + gamma * np.tensordot(probs, V)
                if v > maxV:
                    maxV = v
                    argmaxV = action
            if (argmaxV != old_policy):
                policy[nA, nB] = argmaxV
                policy_stable = False
        if policy_stable:
            return history
        else:
            history.append((np.copy(policy), np.copy(np.round(V, decimals=2))))
            V = policy_evaluation(world, policy, V)

V = np.zeros((max_cars_A + 1, max_cars_B + 1)) # maps state to value
policy = np.zeros((max_cars_A + 1, max_cars_B + 1), dtype='int') # maps state to action (deterministically)
history_initial_problem = policy_improvement(world_initial_problem, actions_set_initial_problem, policy, V)

def plot_history_policy(history, shape=[]):
    """ <TO DO>
    """
    n = len(history)
    h = shape[0] if shape else int(np.sqrt(n))
    w = shape[1] if shape else int(np.ceil(n/h))
    fig, axs = plt.subplots(h, w)
    vmin, vmax = np.inf, - np.inf
    for p, v in history:
        vmin = min(vmin, np.min(p))
        vmax = max(vmax, np.max(p))
    for i in range(len(history)):
        ih = i // w
        iw = i - w * ih
        im = axs[ih, iw].imshow(history[i][0], vmin=vmin, vmax=vmax, origin='lower')
    # ugly handcrafted, but who cares...
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()

#plot_history_policy(history_initial_problem)

# ==========================================================
# NEW PROBLEM ENVIRONMENT
# ==========================================================

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# EXERCISE 4.7 CHANGING PARAMETERS
# Move one car from A to B for free. We can thus move 6 cars from A to B if we want now
# try to have at most 10 cars per location. Otherwise, pay 4 to use parking extension.
free_move_A_to_B = 1

park_restriction_A = 10
park_extension_A = 10 # one extension adds 10 parking spots
park_restriction_B = 10
park_extension_B = 10

new_park_fee = 4
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def actions_set_new_problem(state): # can't move more than max_car_move, or more than we have. Also don't move if other location full
    """ <TO DO>
    """
    nA, nB = state
    return np.arange(-min(nB, max_car_move, max_cars_A - nA), min(nA, max_car_move + free_move_A_to_B, max_cars_B - nB) + 1)

def world_newState_new_problem(NewA, NewB): # no change
    """ <TO DO>
    """
    return world_newState_initial_problem(NewA, NewB)

# compute entire dynamics
print("No need to recompute the world's dynamics during the day, no change there")
probs_from_new_state_new_problem = probs_from_new_state_initial_problem
rent_reward_from_new_state_new_problem = rent_reward_from_new_state_initial_problem

def world_new_problem(state, action): # returns prob for every possible next state and expected reward for this state-action
    """ <TO DO>
    """
    nA, nB = state
    newA, newB = nA - action, nB + action
    probs = probs_from_new_state_new_problem[newA, newB]
    # reward for renting cars
    rent_reward = rent_reward_from_new_state_new_problem[newA, newB]
    # (negative) reward for moving cars at night
    moved_for_free = max(0, min(free_move_A_to_B, action)) # employee moved car for free
    move_reward = - move_fee * (abs(action) - moved_for_free)
    # (negative) reward for using park extensions (in case parameters of problem change)
    park_extensions = ceil(max(0, newA - park_restriction_A) / park_extension_A) + ceil(max(0, newB - park_restriction_B) / park_extension_B)
    park_restriction_reward = - new_park_fee * park_extensions
    # total reward
    reward = rent_reward + move_reward + park_restriction_reward
    return (probs, reward)

# ==========================================================
# REINFORCEMENT LEARNING NEW PROBLEM
# ==========================================================

V = np.zeros((max_cars_A + 1, max_cars_B + 1)) # maps state to value
policy = np.zeros((max_cars_A + 1, max_cars_B + 1), dtype='int') # maps state to action (deterministically)
history_new_problem = policy_improvement(world_new_problem, actions_set_new_problem, policy, V)

#plot_history_policy(history_new_problem)

plot_history_policy(history_initial_problem + history_new_problem)

