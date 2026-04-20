import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory 
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random
import re

# Device configuration
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else: 
    device = "cpu"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Agent:
    def __init__(self, param_set):
        self.param_set = param_set

        # Load parameters from YAML
        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def get_best_reward_from_log(self):
        """Scans the log file to find the highest reward achieved so far."""
        if not os.path.exists(self.LOG_FILE):
            return float("-inf")
        try:
            with open(self.LOG_FILE, "r") as f:
                content = f.read()
                # Find all numbers following 'New Best: '
                rewards = re.findall(r"New Best: ([\d.-]+)", content)
                return max([float(r) for r in rewards]) if rewards else float("-inf")
        except Exception:
            return float("-inf")

    def run(self, is_training=True, render=False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)
        
        # Check if a saved model exists to resume progress
        model_exists = os.path.exists(self.MODEL_FILE)
        if model_exists:
            print(f"--- Loading Checkpoint: {self.MODEL_FILE} ---")
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            
            # Restart Epsilon logic: If resuming, start at 0.15 to use learned knowledge
            # Otherwise, start at 1.0 (epsilon_init)
            epsilon = 0.15 if model_exists else self.epsilon_init
            best_reward = self.get_best_reward_from_log()
            steps_done = 0 
        else:
            if not model_exists:
                print(f"Error: No model found at {self.MODEL_FILE}. Train first!")
                return
            policy_dqn.eval()
            epsilon = 0.0  # Greedy actions only for testing

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            episode_reward = 0
            terminated = False

            while not terminated:
                # Action selection
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).argmax().item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

                # Conversion to tensors
                action_t = torch.tensor(action, dtype=torch.long, device=device)
                reward_t = torch.tensor(reward, dtype=torch.float, device=device)
                next_state_t = torch.tensor(next_state, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action_t, next_state_t, reward_t, terminated))
                    
                    if len(memory) > self.mini_batch_size:
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)
                        
                        steps_done += 1
                        if steps_done % self.network_sync_rate == 0:
                            target_dqn.load_state_dict(policy_dqn.state_dict())

                state = next_state_t
                
                # Stop if threshold reached
                if episode_reward >= self.reward_threshold:
                    break
            
            # Console Output
            status = f"eps={epsilon:.4f}" if is_training else "testing"
            print(f"Episode {episode+1} | Reward: {episode_reward:.2f} | {status}")

            if is_training:
                # Epsilon decay
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                # Save best model
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    with open(self.LOG_FILE, "a") as f:
                        f.write(f"New Best: {best_reward:.2f} at Episode {episode+1}\n")

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        # Bellman Equation
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping (keeps training stable)
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), 1.0)
        
        self.optimizer.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('hyperparameters')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)
    dql.run(is_training=args.train, render=not args.train)