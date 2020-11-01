"""
### NOTICE ###
DO NOT revise this file
"""

import argparse
import numpy as np
from environment import Environment
import time
seed = 11037

def parse():
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project 3")
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def test(agent, env, total_episodes=30):
    rewards = []
    env.seed(seed)
    for i in range(total_episodes):
        state = np.transpose(env.reset(),(2,0,1))
        # agent.init_game_setting()
        done = False
        episode_reward = 0.0
        #playing one game
        print("Test Status: %d"%((i+1)*100/total_episodes),end = '\r')
        y=0
        while(not done):
            # if(i%20) == 0:
            # print(y) 
            # env.render()
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            state = np.transpose(state,(2,0,1))
            episode_reward += reward
            y+=1
            # time.sleep(0.03)

        rewards.append(episode_reward)
    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))
    print("Reward List",rewards)


def run(args):
    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    args = parse()
    run(args)