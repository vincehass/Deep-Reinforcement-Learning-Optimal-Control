import gym
from os.path import join as joindir
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
from agents.Actor_critics.Base_Agent import BaseAgent
from utilities.Actor_Crtics_utilities.PPO.PPO_Network import *
from utilities.Actor_Crtics_utilities.PPO.PPO_helper import *
import matplotlib.pyplot as plt
import datetime
import matplotlib
matplotlib.use('agg')
import seaborn as sns
import pandas as pd
import numpy as np
from utilities.Config import RESULT_DIR



datestr = datetime.datetime.now().strftime('%Y-%m-%d')



class PPOAgent(BaseAgent):
    agent_name = "PPO"
    
    """PPO Agent interacting with environment which is a BaseAgent super class.
    
    Atribute:
    Shared parameters from Base Agent:
    
        env (gym.Env): openAI Gym environment
        
        actor (nn.Module): actor model to select actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic (nn.Module): critic model to predict state values
        critic_optimizer (Optimizer): optimizer for training critic
        
        memory (ReplayBuffer): replay memory
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        device (torch.device): cpu / gpu
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        
        PPO Sub-class of BaseAgent class:
        policy_update_freq (int): policy update frequency
        epsilon (float): amount of clipping surrogate objective
        epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        
          
    """
        
    def __init__(
        
        self,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        ou_noise_theta: float,
        ou_noise_sigma: float,
        epsilon: float,
        epoch: int,
        entropy_weight: float,
        rollout_len: int = 200,
        gamma: float = 0.99,
        tau: float = 5e-3,
        initial_random_steps: int = 1e4,
        policy_update_freq: int = 2,
        exploration_noise: float = 0.1,
        target_policy_noise: float = 0.2,
        target_policy_noise_clip: float = 0.5,
    ):    
        
        super().__init__(            
        env,
        memory_size,
        batch_size,
        gamma,
        tau,
        epsilon,
        epoch,
        rollout_len,
        entropy_weight)
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print('Device used: {}' .format(self.device))
        
       
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        
        
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch = epoch
        self.rollout_len = rollout_len
        self.entropy_weight = entropy_weight
        
        
        
    
        
       
        
        
       # Network
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)
        
        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.005)        
        
        # transition to store in memory
        #self.transition = list()
        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        
        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False
        
        
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            value = self.critic(state)
            self.states.append(state)
            self.actions.append(selected_action)
            self.values.append(value)
            self.log_probs.append(dist.log_prob(selected_action))
         
        return selected_action.cpu().detach().numpy()
    
         
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))     
        
        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, done
               
        
    def update_model(
        self, next_state: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        device = self.device  # for shortening the following lines

        next_state = torch.FloatTensor(next_state).to(device)
        next_value = self.critic(next_state)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.states).view(-1, 3)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values

        actor_losses, critic_losses = [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
            epoch=self.epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        ):
            # calculate ratios
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # actor_loss
            surr_loss = ratio * adv
            clipped_surr_loss = (
                torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
            )
            
            # entropy
            entropy = dist.entropy().mean()

            actor_loss = (
                -torch.min(surr_loss, clipped_surr_loss).mean()
                - entropy * self.entropy_weight
            )

            # critic_loss
            value = self.critic(state)
            #clipped_value = old_value + (value - old_value).clamp(-0.5, 0.5)
            critic_loss = (return_ - value).pow(2).mean()
        
            # train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            
        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []
        
        actor_loss = sum(actor_losses)/ len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

        return actor_loss, critic_loss

    
    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False

        state = self.env.reset()
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses = [], []
        scores = []
        score = 0

        while self.total_step <= num_frames + 1:
            for _ in range(200):
                self.total_step += 1
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward[0][0]
                
                # if episode ends
                if done[0][0]:
                    state = self.env.reset()
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    score = 0

                    self._plot(
                        self.total_step, scores, actor_losses, critic_losses
                    )

            actor_loss, critic_loss = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            print(actor_loss)

        # termination
        self.env.close()

        
    def train_rst(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        
        self.is_test = False

        state = self.env.reset()
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses = [], []
        scores = []
        score = 0

        reward_record = []

       
        while self.total_step in range(1, num_frames + 1):
            for _ in range(300):
                self.total_step += 1
                
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                
                state = next_state
                score += reward[0][0]
                #print(self.total_step, score)
                # if episode ends
                if done[0][0]:
                    state = self.env.reset()
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    score = 0

            actor_loss, critic_loss = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            #print(critic_losses)
                    
            reward_record.append({
            'Agent' : self.agent_name,
            'steps': self.total_step, 
            'Reward': score, 
            'meanepreward': np.mean(scores),
            'stdepreward': np.std(scores)*0.1,
            'Actor loss' : actor_loss,
            'Critic loss' : critic_loss,
            })
                  
            if self.total_step > 1:
            #if self.total_step % plotting_interval == 0:
                print('Finished episode: {} Average Reward: {:.4f} STD Reward: {:.4f} Actor Loss : {:.4f} Critic Loss : {:.4f} ' \
                .format(self.total_step, reward_record[-1]['meanepreward'], reward_record[-1]['stdepreward'],actor_loss, critic_loss))
                print('-----------------')
        
        
        self._plot_record(reward_record)
        
                
        self.env.close()
        
        
        return reward_record
            
            
            
    def _plot_record(
        self, 
        reward_record: List[float], 
            
    ):

        reward_record = pd.DataFrame(reward_record)
        #print(reward_record)
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
        plt.plot(reward_record['steps'], reward_record['Actor loss'], label='loss')
        plt.legend()
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.title('Actor Loss')

        plt.subplot(133)
        plt.plot(reward_record['steps'], reward_record['Critic loss'], label='loss')
        plt.legend()
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.title('Critic Loss')
        
        
        
        
        #plt.show()
        
        reward_record.to_csv(joindir(RESULT_DIR, '{}-record-{}-{}.csv'.format(self.agent_name,self.env.unwrapped.spec.id, datestr)))
        plt.savefig(joindir(RESULT_DIR, '{}-{}-{}.png'.format(self.agent_name,self.env.unwrapped.spec.id, datestr)))
                                
    
    