import argparse
import numpy as np
import pygame
import time
import chess
from chess_env import ChessEnv
from dqn_agent import DQNAgent
import os

def play_game(agent, env, human_mode=False):
    """Play one full game with rendering"""
    state = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return

        env.render()

        if human_mode and env.board.turn == chess.WHITE:  # human plays white
            # Simple human input (click squares) - basic version
            print("Human turn (white). Click two squares or type move (e.g. e2e4)")
            move_str = input("Your move (or 'quit'): ")
            if move_str.lower() == 'quit':
                break
            try:
                move = chess.Move.from_uci(move_str)
                if move in env.board.legal_moves:
                    action = list(env.board.legal_moves).index(move)
                else:
                    print("Illegal move!")
                    continue
            except:
                print("Invalid UCI format!")
                continue
        else:
            # Agent turn
            legal_mask = env.get_legal_moves_mask()
            action = agent.act(state, legal_mask)

        next_state, reward, done, truncated, info = env.step(action)
        done = done or truncated

        total_reward += reward
        state = next_state
        step += 1

        # Slow down for watching
        time.sleep(0.5)

        print(f"Step {step} | Move: {info.get('move', '?')} | Reward: {reward} | Total: {total_reward}")

    print(f"Game over! Result: {env.board.result()} | Total reward: {total_reward}")
    # Save final screenshot
    pygame.image.save(env.screen, "screenshots/final_position.png")
    print("Final board screenshot saved in screenshots/")

def main():
    parser = argparse.ArgumentParser(description="Play with trained Arcade Chess DQN Agent")
    parser.add_argument("--model", default="models/dqn_chess_latest.weights.h5",
                        help="Path to trained model weights")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of games to play")
    parser.add_argument("--human", action="store_true",
                        help="Human plays white (agent black)")
    args = parser.parse_args()

    env = ChessEnv(render_mode="human")
    state_size = env._get_observation().shape[0]
    action_size = env.action_space_size()

    agent = DQNAgent(state_size, action_size)
    if os.path.exists(args.model):
        agent.load(args.model)
        print(f"Loaded model from {args.model}")
        agent.epsilon = 0.0  # no exploration in play mode
    else:
        print(f"Model {args.model} not found! Using random agent.")
        agent.epsilon = 1.0

    for ep in range(1, args.episodes + 1):
        print(f"\nGame {ep}/{args.episodes}")
        play_game(agent, env, human_mode=args.human)

    env.close()

if __name__ == "__main__":
    os.makedirs("screenshots", exist_ok=True)
    main()
