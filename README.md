# DQN-FlappyBird

An autonomous Deep Reinforcement Learning agent built with PyTorch and Gymnasium that learns to master Flappy Bird using a Deep Q-Network (DQN).

## Overview
This project implements a Deep Q-Network to solve the Flappy Bird environment. By utilizing experience replay and a target network, the agent stabilizes the learning process to transition from random flapping to achieving high scores.

## Technical Implementation
* Deep Q-Network (DQN): A neural network consisting of an input layer for game states, a hidden layer of 256 neurons, and an output layer for actions.
* Experience Replay: Implements a circular buffer (ReplayMemory) to store 100,000 transitions, allowing the agent to learn from non-sequential past experiences.
* Target Network Synchronization: Uses a separate network to calculate stable Q-targets, which is synchronized every 10 steps to prevent training divergence.
* Stability Measures: Incorporates MSE Loss for optimization and Gradient Norm Clipping to keep training updates stable.
* Hardware Acceleration: Automatically detects and utilizes CUDA or Apple’s MPS for high-performance training.

## Hyperparameters
Configurations are managed via a centralized parameters.yaml file:
* Alpha (Learning Rate): 0.0001
* Gamma (Discount Factor): 0.99
* Batch Size: 32
* Epsilon Decay: 0.999

## Project Structure
* Agent.py: Contains the training loop, action selection logic, and the optimize function.
* dqn.py: Defines the PyTorch neural network architecture.
* experience_replay.py: Manages the storage and sampling of the agent's memory.
* parameters.yaml: Stores all environment and training hyperparameters.
* game_flappy_bird.py: Script to manually test the environment via keyboard input.

## Getting Started

### Installation
pip install gymnasium flappy-bird-gymnasium torch pyyaml pygame

### Usage
To train the agent:
python Agent.py flappybirdv0 --train

To watch the trained agent play:
python Agent.py flappybirdv0

## Results
The agent tracks performance in real-time, logging new best scores and saving model checkpoints to the runs/ directory.

