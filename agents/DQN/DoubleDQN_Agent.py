from os import environ
from matplotlib import markers
import os
import sys
from os.path import join as joindir
import torch
import torch.nn.functional as F
from agents.DQN.Base_agent_DQN import BaseAgent
import gym
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import matplotlib
matplotlib.use('agg')




datestr = datetime.datetime.now().strftime('%Y-%m-%d')


class DoubleDQNAgent(BaseAgent):
    agent_name = "Double_DQN"
    
    """DQN Agent interacting with environment which is a BaseAgent super class.
    
    Attribute:
        
        Shared parameters:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        
        DQN parameters:
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
    """
    def __init__(
        
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        prioritized_memory: bool,
        # N-step Learning
        n_step: int = 1,
        use_n_step: bool = False,
    ):
        super().__init__( 
        
        env,
        memory_size,
        batch_size,
        target_update,
        epsilon_decay,
        prioritized_memory,
        n_step,
        use_n_step,
        )
        
        

    def update_model(self) -> torch.Tensor:
            """Update the model by gradient descent."""
            samples = self.memory.sample_batch()

            loss = self._compute_double_dqn_loss(samples)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()
            
        
        
    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False
        
        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # if episode ends
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                
                # linearly decrease epsilon
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses, epsilons)
                
        self.env.close()
                
    def test(self, video_folder: str) -> None:
        """Test the agent."""
        self.is_test = True
        
        # for recording a video
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)
        
        state = self.env.reset()
        done = False
        score = 0
        
        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()
        
        # reset
        self.env = naive_env

    def _compute_double_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).gather(
            1, self.dqn(next_state).argmax(dim=1, keepdim=True)  # Double DQN
        ).detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    def _plot(
        self, 
        #frame_idx: int, 
        num_frames: int, 
        scores: List[float], 
        losses: List[float], 
        epsilons: List[float],
    ):
        """Plot the training progresses."""
        
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        #plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.title('frames %s. score: %s' % (num_frames, np.mean(scores[-10:])))
        plt.plot(scores, linestyle='--', marker='o', color='b')
        # plt.fill_between(scores-np.std(scores),scores-np.std(scores),
        #          color='gray', alpha=0.2)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()        
        
        
        
    def _plot_record(
        self, 
        reward_record: List[float], 
            
    ):
        
        #record_dfs = pd.DataFrame(columns=['steps', 'reward'])
        reward_record = pd.DataFrame(reward_record)
        reward_record['reward_smooth'] = reward_record['meanepreward'].ewm(span=200).mean()
        """Plot the training progresses."""        
        
        
        sns.set_theme()
        sns.set_style("darkgrid")
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.plot(reward_record['steps'], reward_record['meanepreward'], label='trajectory reward')
        plt.plot(reward_record['steps'], reward_record['reward_smooth'], label='smoothed reward')
        plt.fill_between(reward_record['steps'], reward_record['meanepreward'] - reward_record['stdepreward'], 
            reward_record['meanepreward'] + reward_record['stdepreward'], color='b', alpha=0.2)
        plt.legend()
        plt.xlabel('steps of env interaction (sample complexity)')
        plt.ylabel('average reward')
        plt.title('{} on {}'.format(self.agent_name,self.env.unwrapped.spec.id))
        
        
        plt.subplot(132)
        plt.plot(reward_record['steps'], reward_record['epsilons_var'], label='epsilon sensitivity')
        #plt.plot(reward_record['steps'], reward_record['epsilons'], label='epsilon')
        plt.legend()
        plt.xlabel('steps')
        plt.ylabel('epsilon variation')
        plt.title('Epsilon Decay')
        
        # plt.subplot(132)
        # plt.plot(reward_record['steps'], reward_record['epsilons'], label='epsilon')
        # plt.legend()
        # plt.xlabel('steps')
        # plt.ylabel('epsilon')
        # plt.title('Epsilon Decay')
        
        plt.subplot(133)
        plt.plot(reward_record['steps'], reward_record['losses'], label='loss')
        plt.legend()
        plt.xlabel('steps')
        plt.ylabel('average loss')
        plt.title('Loss')
        
        
        
        #plt.show()
        
        reward_record.to_csv(joindir(RESULT_DIR, '{}-record-{}-{}.csv'.format(self.agent_name,self.env.unwrapped.spec.id, datestr)))
        plt.savefig(joindir(RESULT_DIR, '{}-{}-{}.pdf'.format(self.agent_name,self.env.unwrapped.spec.id, datestr)))
    
    def train_rst(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False
        
        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0
        reward_record = []
        len_list = []

        
        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # if episode ends
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                len_list.append(update_cnt + 1)
                
                # linearly decrease epsilon
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()
                
                reward_record.append({
                'Agent' : self.agent_name,
                'Episode': frame_idx, 
                'steps': update_cnt, 
                'Reward': score, 
                'meanepreward': np.mean(scores),
                'stdepreward': np.std(scores)*0.1,
                'epsilons' : self.epsilon,
                'epsilons_var' : 100.0*self.epsilon/np.mean(loss),
                'losses' : loss,
                'meaneplen': np.mean(len_list)})
        
                if frame_idx % self.target_update == 0:
                    print('Finished episode: {} Average Reward: {:.4f} STD Reward: {:.4f} Total Loss = {:.4f} epsilon = {:.1f}% Buffer size = {:}' \
                .format(frame_idx, reward_record[-1]['meanepreward'], reward_record[-1]['stdepreward'],loss, reward_record[-1]['epsilons_var'], self.memory.size))
                    print('-----------------')
            
            
        # # plotting inside the loop
        # if frame_idx % plotting_interval == 0:
        #     self._plot(num_frames, scores, losses, epsilons)
        #self._plot(num_frames, scores, losses, epsilons)
        
        
        self._plot_record(reward_record)
        
                
        self.env.close()
        
        
        return reward_record
        
        
    