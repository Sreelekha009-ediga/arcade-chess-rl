from chess_env import ChessEnv
from dqn_agent import DQNAgent

env = ChessEnv()
state_size = env._get_observation().shape[0]
action_size = 4672  # max possible

agent = DQNAgent(state_size, action_size)
print("Agent created successfully!")
print("Epsilon:", agent.epsilon)

state = env.reset()
action = agent.act(state)
print("Sample action:", action)

env.close()