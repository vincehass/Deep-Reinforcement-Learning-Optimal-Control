import argparse
from email.policy import default
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
import matplotlib.pyplot as plt
matplotlib.use('agg')
import datetime
from agents.Actor_critics.DDPG_Agent import DDPGAgent
from agents.Actor_critics.SACAgent import SACAgent
from agents.Actor_critics.PPO_Agent import PPOAgent
from environements.Action_normalizer import ActionNormalizer
from utilities.Config import RESULT_DIR

datestr = datetime.datetime.now().strftime('%Y-%m-%d')



# hyper-parameters

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'Acrtor Crtici Agent and its variants')
    parser.add_argument('--num_frames', type = int, default = 50000, help = 'number of frames')
    parser.add_argument('--env_id', type = str, default = 'Pendulum-v1', help = 'environement')
    parser.add_argument('--memory_size', type = int, default = 10000, help = 'memory size')
    parser.add_argument('--batch_size', type = int, default = 128, help = 'batch size')
    parser.add_argument('--gamma', type = float, default = 0.99, help = 'gamma')
    parser.add_argument('--tau', type = float, default = 5e-3, help = 'tau')
    parser.add_argument('--ou_noise_theta', type = float, default = 1.0, help = 'ou_noise theta')
    parser.add_argument('--ou_noise_sigma', type = float, default = 0.1, help = 'ou_noise sigma')
    parser.add_argument('--initial_random_steps', type = int, default = 1e4, help = 'initial random steps')
    parser.add_argument('--epsilon', type = float, default = 0.2, help = 'epsilon')
    parser.add_argument('--epoch', type = int, default = 64, help = 'epoch')
    parser.add_argument('--rollout_len', type = int, default = 2048, help = 'rollout_len')
    parser.add_argument('--policy_update_freq', type = int, default = 2, help = 'policy_update_freq')
    parser.add_argument('--exploration_noise', type = float, default = 0.1, help = 'exploration_noise')
    parser.add_argument('--target_policy_noise', type = float, default = 0.2, help = 'target_policy_noise')
    parser.add_argument('--target_policy_noise_clip', type = float, default = 0.5, help = 'target_policy_noise_clip')    
    parser.add_argument('--entropy_weight', type = float, default = 0.05, help = 'enropy_weight')    
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
        print('Agent {} on Environement {}'.format(i.agent_name, args.env_id))               
        agent = i(args.env, args.memory_size, args.batch_size,args.ou_noise_theta, args.ou_noise_sigma,
                    args.epsilon, args.epoch, args.rollout_len, args.entropy_weight,
                        args.gamma, args.tau, args.initial_random_steps, args.policy_update_freq,
                            args.exploration_noise, args.target_policy_noise, args.target_policy_noise_clip
                )
        
        
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
    import seaborn as sns
    sns.set_theme()
    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data = record_dfs, x = 'steps', y = 'meanepreward', hue = 'Agent')#, style ='meanepreward' )
    #plt.legend()
    plt.xlabel('Steps of env interaction (sample complexity)')
    plt.ylabel('Average Reward')
    plt.title('Actor Critics Agents and its variants on {}'.format(args.env_id))
    plt.savefig(joindir(RESULT_DIR, 'Actor Critics-Agents-records on {}-{}.png'.format(args.env_id, datestr)))
                  

def main():

    # environment
    args.env_id = 'Pendulum-v1'
    args.env = gym.make(args.env_id)
    args.env = ActionNormalizer(args.env)
    np.random.seed(args.seed)    
    seed_torch(args.seed)
    args.env.seed(args.seed)
    
    # Agents = [SACAgent,DDPGAgent, PPOAgent] 
    Agents = [PPOAgent]
    global_plot(Agents, args)
    
if __name__ == "__main__":
    main()
    
    
    
    

        
        
        