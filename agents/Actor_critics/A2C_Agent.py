import os
import sys
from os.path import join as joindir
import gym
import torch
import torch.nn.functional as F
from typing import List, Tuple
from agents.Actor_critics.A2C_Agent import BaseAgent
import matplotlib.pyplot as plt
import datetime
import matplotlib
matplotlib.use('agg')
import seaborn as sns
import pandas as pd
import numpy as np



datestr = datetime.datetime.now().strftime('%Y-%m-%d')



class A2CAgent(BaseAgent):
    agent_name = "A2C"
    
    """A2C Agent interacting with environment which is a BaseAgent super class.
    
    Atribute:
    Shared parameters:
    
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """
    def __init__(
        
        self, env: gym.Env, gamma: float, entropy_weight: float
    ):    
        
        super().__init__(env, gamma, entropy_weight)
        
        
        
        

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
            """Update the model by gradient descent."""  
            state, log_prob, next_state, reward, done = self.transition

            # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
            #       = r                       otherwise
            mask = 1 - done
            next_state = torch.FloatTensor(next_state).to(self.device)
            pred_value = self.critic(state)
            targ_value = reward + self.gamma * self.critic(next_state) * mask
            value_loss = F.smooth_l1_loss(pred_value, targ_value.detach())
            
            # update value
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

            # advantage = Q_t - V(s_t)
            advantage = (targ_value - pred_value).detach()  # not backpropagated
            policy_loss = -advantage * log_prob
            policy_loss += self.entropy_weight * -log_prob  # entropy maximization

            # update policy
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            return policy_loss.item(), value_loss.item()
    
    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False
        
        actor_losses, critic_losses, scores = [], [], []
        state = self.env.reset()
        score = 0
        
        for self.total_step in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            
            actor_loss, critic_loss = self.update_model()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            
            state = next_state
            score += reward
            
            # if episode ends
            if done:         
                state = self.env.reset()
                scores.append(score)
                score = 0                
            
            # plot
            if self.total_step % plotting_interval == 0:
                print('Finished episode: {} Average Reward: {:.4f} Actor Loss = {:.4f} Critic Loss = {:.4f} ' \
                .format(self.total_step, score, actor_loss,  critic_loss))
                print('-----------------')
                self._plot(self.total_step, scores, actor_losses, critic_losses)
        self.env.close()
    
    def test(self):
        """Test the agent."""
        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        
        frames = []
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()
        
        return frames
    
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        actor_losses: List[float], 
        critic_losses: List[float], 
    ):
        """Plot the training progresses."""
        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (131, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores),
            (132, "actor_loss", actor_losses),
            (133, "critic_loss", critic_losses),
        ]

        plt.figure(figsize=(30, 5))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
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
        plt.plot(reward_record['steps'], reward_record['actor_loss'], label='Actor loss')
        #plt.plot(reward_record['steps'], reward_record['epsilons'], label='epsilon')
        plt.legend()
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.title('Actor Loss')

        # plt.subplot(132)
        # plt.plot(reward_record['steps'], reward_record['epsilons'], label='epsilon')
        # plt.legend()
        # plt.xlabel('steps')
        # plt.ylabel('epsilon')
        # plt.title('Epsilon Decay')

        plt.subplot(133)
        plt.plot(reward_record['steps'], reward_record['critic_loss'], label='Critic loss')
        #plt.plot(reward_record['steps'], reward_record['epsilons'], label='epsilon')
        plt.legend()
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.title('Critic Loss')



        #plt.show()

        reward_record.to_csv(joindir(RESULT_DIR, '{}-record-{}-{}.csv'.format(self.agent_name,self.env.unwrapped.spec.id, datestr)))
        plt.savefig(joindir(RESULT_DIR, '{}-{}-{}.pdf'.format(self.agent_name,self.env.unwrapped.spec.id, datestr)))



    def train_rst(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False
        
        actor_losses, critic_losses, scores = [], [], []
        state = self.env.reset()
        score = 0
        reward_record = []
        len_list = []
        scores = []

        
        for self.total_step in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            
            
            
            actor_loss, critic_loss = self.update_model()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            
            state = next_state
            score += reward
            
            
            # if episode ends
            if done:         
                state = self.env.reset()
                scores.append(score)
                score = 0     
                
                # # if hard update is needed
                # if update_cnt % self.target_update == 0:
                #     self._target_hard_update()
                
                reward_record.append({
                'Agent' : self.agent_name,
                'steps': self.total_step, 
                'Reward': score, 
                'meanepreward': np.mean(scores),
                'stdepreward': np.std(scores)*0.1,
                'actor_loss' : actor_loss,
                'critic_loss' : critic_loss,
                'meaneplen': np.mean(len_list)})
        
                if self.total_step % plotting_interval == 0:
                    print('Finished episode: {} Average Reward: {:.4f} STD Reward: {:.4f} Actor Loss = {:.4f} Critic Loss = {:.4f} ' \
                .format(self.total_step, reward_record[-1]['meanepreward'], reward_record[-1]['stdepreward'], actor_loss,  critic_loss))
                    print('-----------------')
            
                # # plot
                # if self.total_step % plotting_interval == 0:
                #     self._plot(self.total_step, scores, actor_losses, critic_losses)
            
        self.env.close()                
        
        
        #self._plot_record(reward_record)
        
                
        
        
        
        return reward_record