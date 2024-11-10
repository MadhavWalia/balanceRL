import numpy as np
import random
from environment.load_balancer import LoadBalancerState
from .base import BaseAgent

class DynaQAgent(BaseAgent):
    def __init__(self, n_servers: int, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.1, 
                 planning_steps: int = 5):
        super().__init__(n_servers)
        
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        self.model = {}  # Model to store (next_state, reward) for (state, action) pairs
        self.planning_steps = planning_steps  # Number of simulated updates per real experience

    def _state_to_key(self, state: LoadBalancerState) -> tuple:
        return tuple(np.round(state.server_loads, 2))

    def select_action(self, state: LoadBalancerState) -> int:
        state_key = self._state_to_key(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_servers)
        
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_servers)
        
        return np.argmax(self.q_table[state_key])

    def update(self, state: LoadBalancerState, action: int, 
               reward: float, next_state: LoadBalancerState, done: bool):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # Q-Learning update for the actual experience
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.n_servers)
        
        current_q = self.q_table[state_key][action]
        next_max_q = 0 if done else np.max(self.q_table[next_state_key])
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][action] = new_q
        
        # Store the experience in the model
        self.model[(state_key, action)] = (reward, next_state_key)

        # Planning step: simulate additional updates using the stored model
        for _ in range(self.planning_steps):
            simulated_state_action = random.choice(list(self.model.keys()))
            simulated_reward, simulated_next_state_key = self.model[simulated_state_action]
            
            sim_state_key, sim_action = simulated_state_action
            sim_next_max_q = np.max(self.q_table.get(simulated_next_state_key, np.zeros(self.n_servers)))
            
            # Q-update for simulated experience
            self.q_table[sim_state_key][sim_action] += self.lr * (
                simulated_reward + self.gamma * sim_next_max_q - self.q_table[sim_state_key][sim_action]
            )

