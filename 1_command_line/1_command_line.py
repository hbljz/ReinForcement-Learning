#-*-coding:utf8-*-
__author__ = '何斌'

import numpy as np
import pandas as pd
import time
import openpyxl
import os

# np.random.seed(2)  # reproducible

N_STATES = 6                    # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9                   # greedy police(取Q表的百分比)
ALPHA = 0.1                     # learning rate
GAMMA = 0.9                     # discount factor(奖励衰减值)
MAX_EPISODES = 13               # maximum episodes(局数)
FRESH_TIME = 0.1                # fresh time for one move
read_save=False                 #是否保存Q_table


def build_q_table(n_states, actions):
    global read_save
    if read_save and os.path.exists("q_table.xlsx"):
        table = pd.read_excel("q_table.xlsx", 'Sheet1')
    else:
        table = pd.DataFrame(
            np.zeros((n_states, len(actions))),   # q_table initial values
            columns=actions,   					  # actions's name
        )
    # print(table) show table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.argmax()
    return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    # This is how environment be updated
    print('\n'*50)
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print(interaction,"\n")
        time.sleep(2)
    else:
        env_list[S] = 'o'
        interaction = ' '.join(env_list)
        # print('\r{}'.format(interaction), end='')
        print(interaction)
        time.sleep(FRESH_TIME)

def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        #保存Q_table
        if read_save:
            q_table.to_excel("q_table.xlsx", sheet_name='Sheet1')

        #初始化状态    
        update_env(S, episode, step_counter) 
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            q_predict = q_table.ix[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.ix[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next stated
            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)