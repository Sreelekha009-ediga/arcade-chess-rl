# Arcade Chess RL

**Self-learning chess agent using Deep Q-Learning (DQN) + Pygame visualization**


### Key Highlights 
- Designed a self-learning chess agent using Deep Q-Learning (DQN).
- Defined reward mechanisms for captures (+2), checks (+1.5), and checkmate (+50) to improve learning efficiency.
- Visualized agent behavior in real time using Pygame for debugging and optimization.
- Tech Stack: Python, python-chess, Pygame, TensorFlow/Keras, Reinforcement Learning

### Project Structure
arcade-chess-rl/
├── main.py          # Play game (human vs agent or agent self-play)
├── chess_env.py     # Gym-like environment (state, step, reward, render)
├── dqn_agent.py     # DQN model + replay buffer
├── train.py         # Training loop (self-play, experience replay)
├── utils.py         # (optional – add later if needed)
├── requirements.txt
├── models/          # Saved weights (dqn_chess_latest.weights.h5)
├── assets/          # Chess piece PNGs
└── screenshots/     # Game screenshots


### How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Train the agent :python train.py
3. Watch agent play:python main.py
4. Play against the agent (you as White):python main.py --human