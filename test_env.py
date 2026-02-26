import pygame
from chess_env import ChessEnv

env = ChessEnv(render_mode="human")
obs = env.reset()
print("Observation shape:", obs.shape)
print("Board:\n", env.board)
env.render()
pygame.time.wait(3000)  # show for 3 seconds
env.close()