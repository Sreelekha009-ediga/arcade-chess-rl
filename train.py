import numpy as np
from chess_env import ChessEnv
from dqn_agent import DQNAgent
import random
import time
import os

# Hyperparameters
EPISODES = 250        # start small, increase to 5000+ later
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 100  # update target network every X episodes
SAVE_MODEL_EVERY = 500
MAX_STEPS_PER_GAME = 200  # prevent infinite games

def train():
    env = ChessEnv(render_mode=None)  # no render during training (fast)
    state_size = env._get_observation().shape[0]
    action_size = env.action_space_size()

    agent = DQNAgent(state_size, action_size)

    # Optional: load previous model if exists
    if os.path.exists("models/dqn_chess_latest.weights.h5"):
        agent.load("models/dqn_chess_latest.weights.h5")
        print("Loaded previous model")

    episode_rewards = []
    avg_rewards = []

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < MAX_STEPS_PER_GAME:
            legal_mask = env.get_legal_moves_mask()
            action = agent.act(state, legal_mask)

            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            # Train on mini-batch every 4 steps
            if len(agent.memory) > BATCH_SIZE and steps % 4 == 0:
                agent.replay(BATCH_SIZE)

        # Update target network periodically
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()

        episode_rewards.append(total_reward)
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        avg_rewards.append(avg_reward)

        print(f"Episode {episode}/{EPISODES} | Reward: {total_reward:.2f} | Avg(100): {avg_reward:.2f} | "
              f"Epsilon: {agent.epsilon:.3f} | Steps: {steps} | Result: {env.board.result()}")

        # Save model periodically
        if episode % SAVE_MODEL_EVERY == 0:
            agent.save(f"models/dqn_chess_ep{episode}.weights.h5")
            agent.save("models/dqn_chess_latest.weights.h5")
            print(f"Model saved at episode {episode}")

    env.close()

    # Final save
    agent.save("models/dqn_chess_final.weights.h5")
    print("Training complete! Final model saved.")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train()