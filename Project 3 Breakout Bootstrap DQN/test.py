"""
### NOTICE ###
DO NOT revise this file
"""

import argparse
import numpy as np
from environment import Environment
import time
import torch
seed = 11037
from torch.utils.tensorboard import SummaryWriter
Path_test_weights = './max_test_weights_bootdqn_3.tar'
tensor_board_dir='./logs/test_data'
#writer = SummaryWriter(tensor_board_dir) 
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
    game_reward_list = []
    max_game_reward=0
    per_game_reward = 0
    # env.seed(seed)
    for i in range(total_episodes):
        state = np.transpose(env.reset(),(2,0,1))
        # agent.init_game_setting()
        done = False
        episode_reward = 0.0
        start = True
        #playing one game
        print("Test Status: %d"%((i+1)*100/total_episodes),end = '\r')
        y=0
        c=0
        d=1
        terminal = False
        while not done:
            # if(i%20) == 0:
            # time.pause(0.01) 
            action = agent.make_action(state, test=True)
            # if y>(2000*d):
            #     # print("Using Random Action, y = ",y)
            #     action = np.random.randint(0,4)
                # if y==(2000*d+15):
                #     d+=1
            state, reward, done, info = env.step(action)
            # if info['ale.lives'] == 0:
            #     per_game_reward = 0
            # env.reset()
            # env.render()
            if info['ale.lives'] == 5 and start:
                start = False
                if i>0:
                    game_reward_list.append(per_game_reward)
                    print("Game Number: %d, Reward: %d"%(len(game_reward_list),per_game_reward))
                    max_game_reward = per_game_reward if per_game_reward>max_game_reward else max_game_reward
                    per_game_reward = 0
            # if info['ale.lives'] == 0:
            #     print(per_game_reward)
            state = np.transpose(state,(2,0,1))
            episode_reward += reward
            # print(y,done,info,episode_reward)
            # print(y)
            per_game_reward +=reward
            y+=1
            # time.sleep(0.1)
        # episode_reward = 0
        rewards.append(episode_reward)

        # rewards.append(episode_reward)
    print('Run %d episodes'%(len(rewards)))
    print('Episodic_Mean:', np.mean(rewards))
    print("Reward List",rewards)
    print('Played %d Games'%(len(game_reward_list)))
    print('Game_Mean:', np.mean(game_reward_list))
    print('Max_game_reward:', max_game_reward)
    #writer.add_scalar("Mean_test_reward(100 Episodes)",np.mean(rewards),agent.eval_num)
    # if np.mean(rewards)>=agent.max_test_reward:
    #   agent.max_test_reward = np.mean(rewards)
    #   print("Max_Reward = %0.2f"%agent.max_test_reward)
    #   print("Saving_Test_Weights_Model")
    #   torch.save({
    #     'target_state_dict':agent.Target_DQN.state_dict(),
    #     'train_state_dict':agent.DQN.state_dict(),
    #     'optimiser_state_dict':agent.optimiser.state_dict()
    #     },Path_test_weights)

def run(args):
    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    args = parse()
    run(args)
