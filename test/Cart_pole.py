import argparse
import os
import sys
from os.path import join as joindir
from os.path import dirname, abspath

sys.path.append(dirname(dirname(abspath(__file__))))

import gym
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import datetime
from agents.DQN.DQN_Agent import DQNAgent
from agents.DQN.DoubleDQN_Agent import DoubleDQNAgent
from agents.DQN.DQNNStepAgent import DQNNStepAgent
from agents.DQN.DQN_PER import DQNPERAgent
#from environements.Action_normalizer import ActionNormalizer



datestr = datetime.datetime.now().strftime('%Y-%m-%d')



# parameters

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'DQN Agent and its variants')
    parser.add_argument('--env_id', type = str, default = 'CartPole-v1', help = 'environement')
    parser.add_argument('--num_frames', type = int, default = 10000, help = 'number of frames')
    parser.add_argument('--memory_size', type = int, default = 1000, help = 'memory size')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size')
    parser.add_argument('--target_update', type = int, default = 100, help = 'target update')
    parser.add_argument('--epsilon_decay', type = int, default = 1/2000, help = 'epsilon decay')
    parser.add_argument('--prioritized_memory', type = bool, default = False, help = 'prioritized_memory')
    parser.add_argument('--n_step', type = int, default = 1, help = 'n_step')
    parser.add_argument('--num_parallel_run', type = int, default = 3, help = 'num parallel run')
    parser.add_argument('--seed', type = int, default = 123, help = 'seed')
    
    args = parser.parse_args()
    
    return args

args = parse_arguments()   


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def global_plot(Agents, args):
        
    reward_cols = []
    appreward_record_append = []
    record_dfs = pd.DataFrame(columns=['steps', 'Reward'])
    
    for i in Agents:
        args.seed += 1
        # if(i == DQNNStepAgent):
        #     args.n_step = 3
        # else:
        #     args.n_step = 1
        print('Agent {} on Environement {}'.format(i.agent_name, args.env_id))               
        agent = i(args.env, args.memory_size, args.batch_size, args.target_update, args.epsilon_decay, 
                  args.prioritized_memory, args.n_step)
        
        
        rollout = agent.train_rst(args.num_frames)
        reward_record = pd.DataFrame(rollout)
        appreward_record_append.append(reward_record)

    appreward_record_append = pd.concat(appreward_record_append)    
    appreward_record_append.to_csv(joindir(RESULT_DIR, '{}-records-{}-{}.csv'.format(str(Agents[0].agent_name),args.env_id, datestr)))
    record_dfs = record_dfs.merge(appreward_record_append, how='outer', on='steps', suffixes=('', '_{}'.format(i.agent_name)))
    reward_cols.append('Reward_{}'.format(i.agent_name))

    record_dfs = record_dfs.drop(columns='Reward').sort_values(by='steps', ascending=True).ffill().bfill()

    record_dfs['reward_mean'] = record_dfs[reward_cols].mean(axis=1)
    record_dfs['reward_std'] = record_dfs[reward_cols].std(axis=1)
    record_dfs['reward_smooth'] = record_dfs['reward_mean'].ewm(span=20).mean()
    record_dfs.to_csv(joindir(RESULT_DIR, '{}-Agents-record-{}-{}.csv'.format(str(Agents[0].agent_name),args.env_id, datestr)))

    #Plot
    
    sns.set_theme()
    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data = record_dfs, x = 'steps', y = 'meanepreward', hue = 'Agent')#, style ='meanepreward' )
    plt.xlabel('Steps of env interaction (sample complexity)')
    plt.ylabel('Average Reward')
    plt.title('DQN Agents and its variants on {}'.format(args.env_id))
    plt.savefig(joindir(RESULT_DIR, 'DQN-Agents-records on {}-{}.pdf'.format(args.env_id, datestr)))
                  

def main():

    # environment
    args.env_id = 'CartPole-v1'
    args.env = gym.make(args.env_id)
    np.random.seed(args.seed)    
    seed_torch(args.seed)
    args.env.seed(args.seed)
    
    Agents = [DQNAgent, DoubleDQNAgent, DQNPERAgent, DQNNStepAgent]
    global_plot(Agents, args)                  
    
if __name__ == "__main__":
    main()