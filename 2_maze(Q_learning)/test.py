#-*-coding:utf8-*-
__author__ = '何斌'

from maze_env import Maze
from RL_brain import QLearningTable
import openpyxl

def update():
    for episode in range(50):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                #保存q_table时使用
                RL.save_q_table()
                break

    # end of game
    print('game over')
    print(RL.q_table)
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)),read_save=False)
    env.after(100, update)
    env.mainloop()