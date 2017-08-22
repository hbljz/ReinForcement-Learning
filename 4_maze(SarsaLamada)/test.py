#-*-coding:utf8-*-
__author__ = '何斌'

from maze_env import Maze
from RL_brain import SarsaLambdaTable
import openpyxl

read_save=True     #是否保存Q_table,或读取已保存的Q_table

def update():
    for episode in range(20):
        # initial observation
        observation = env.reset()

        # RL choose action based on observation
        action = RL.choose_action(str(observation))
        
        # initial all zero eligibility trace
        RL.eligibility_trace *= 0

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                #保存q_table时使用
                if RL.read_save:
                    RL.save_q_table()
                break

    # end of game
    print('game over')
    print("Q_table:\n",RL.q_table)
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)),read_save=read_save)
    env.after(100, update)
    env.mainloop()