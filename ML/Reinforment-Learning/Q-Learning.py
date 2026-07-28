# Q Learning
"""
This code sets up a simple RL environment and uses the Q-learning algorithm to teach an agent
how to navigate through it. The agent learns to maximize its total reward by avoiding obstacles
and reaching the goal in the fewest steps possible. Through exploration and exploitation, 
the agent improves its strategy over episodes, as reflected in the updating Q-table values.
"""
import numpy as np

# Environment setup
grid_size = 5
goal_state = (4, 4)
obstacle_state = (2, 2)
actions = {'up': -1, 'down': 1, 'left': -1, 'right': 1}

# Initialize Q-table
q_table = np.zeros((grid_size, grid_size, len(actions)))

# Hyperparameters
learning_rate = 0.1
discount_rate = 0.99
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.01
total_episodes = 1000

# Helper function to choose action
def choose_action(state):
    if np.random.uniform(0, 1) < exploration_rate:
        action = np.random.choice(list(actions.keys()))  # Explore action space
    else:
        action_index = np.argmax(q_table[state[0], state[1], :])  # Exploit learned values
        action = list(actions.keys())[action_index]
    return action

# Helper function to update the state
def update_state(state, action):
    if action == 'up' and state[0] > 0:
        return (state[0] + actions[action], state[1])
    elif action == 'down' and state[0] < grid_size - 1:
        return (state[0] + actions[action], state[1])
    elif action == 'left' and state[1] > 0:
        return (state[0], state[1] + actions[action])
    elif action == 'right' and state[1] < grid_size - 1:
        return (state[0], state[1] + actions[action])
    return state  # Return the same state if no valid move

# Q-learning algorithm
for episode in range(total_episodes):
    state = (0, 0)
    done = False
    
    while not done:
        action = choose_action(state)
        new_state = update_state(state, action)
        reward = -1  # Default reward
        
        if new_state == goal_state:
            reward = 100
            done = True
        elif new_state == obstacle_state:
            reward = -100
            done = True
        
        action_index = list(actions.keys()).index(action)
        # Q-table update
        q_table[state[0], state[1], action_index] = q_table[state[0], state[1], action_index] + learning_rate * (reward + discount_rate * np.max(q_table[new_state[0], new_state[1], :]) - q_table[state[0], state[1], action_index])
        
        state = new_state
        
    # Exploration rate decay
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

print("Training completed.")