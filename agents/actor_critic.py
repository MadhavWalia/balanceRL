import numpy as np
from environment.load_balancer import LoadBalancerState
from .base import BaseAgent

class ActorCriticAgent(BaseAgent):
    def __init__(self, n_servers: int, actor_lr: float = 0.1, 
                 critic_lr: float = 0.1, discount_factor: float = 0.95, 
                 epsilon: float = 0.1):
        super().__init__(n_servers)
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.actor_table = {}  # Policy table (actor)
        self.critic_table = {}  # Value function table (critic)
    
    def _state_to_key(self, state: LoadBalancerState) -> tuple:
        return tuple(np.round(state.server_loads, 2))
    
    def select_action(self, state: LoadBalancerState) -> int:
        state_key = self._state_to_key(state)
        
        if state_key not in self.actor_table:
            self.actor_table[state_key] = np.ones(self.n_servers) / self.n_servers  # Uniform policy
        
        # Epsilon-greedy policy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_servers)
        
        # Choose action based on policy
        action_probabilities = self.actor_table[state_key]
        return np.random.choice(np.arange(self.n_servers), p=action_probabilities)
    
    def update(self, state: LoadBalancerState, action: int, 
               reward: float, next_state: LoadBalancerState, done: bool):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # Initialize state-value if unseen
        if state_key not in self.critic_table:
            self.critic_table[state_key] = 0.0
        if next_state_key not in self.critic_table:
            self.critic_table[next_state_key] = 0.0
        if state_key not in self.actor_table:
            self.actor_table[state_key] = np.ones(self.n_servers) / self.n_servers
        
        # Calculate TD error
        current_value = self.critic_table[state_key]
        next_value = 0 if done else self.critic_table[next_state_key]
        td_error = reward + self.gamma * next_value - current_value
        
        # Critic update
        self.critic_table[state_key] += self.critic_lr * td_error
        
        # Actor update (policy gradient)
        policy = self.actor_table[state_key]
        policy_gradient = np.zeros(self.n_servers)
        policy_gradient[action] = 1  # One-hot encoding for the taken action
        policy_gradient -= policy  # Baseline correction
        
        # Update policy using TD error as a weight for the gradient
        self.actor_table[state_key] += self.actor_lr * td_error * policy_gradient
        self.actor_table[state_key] = np.clip(self.actor_table[state_key], 0, 1)  # Keep within bounds
        self.actor_table[state_key] /= np.sum(self.actor_table[state_key])  # Normalize
