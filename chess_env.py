import chess
import numpy as np
import pygame
from typing import Tuple, Dict, Any

class ChessEnv:
    def __init__(self, render_mode: str = None):
        self.board = chess.Board()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.piece_images = {}
        self.square_size = 64  # pixels per square

        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((8 * self.square_size, 8 * self.square_size))
            pygame.display.set_caption("Arcade Chess RL")
            self.clock = pygame.time.Clock()
            self._load_pieces()

    def _load_pieces(self):
        """Load piece images from assets/"""
        pieces = {
            'K': 'w_king.png', 'Q': 'w_queen.png', 'R': 'w_rook.png',
            'B': 'w_bishop.png', 'N': 'w_knight.png', 'P': 'w_pawn.png',
            'k': 'b_king.png', 'q': 'b_queen.png', 'r': 'b_rook.png',
            'b': 'b_bishop.png', 'n': 'b_knight.png', 'p': 'b_pawn.png',
        }
        for symbol, filename in pieces.items():
            try:
                img = pygame.image.load(f"assets/{filename}")
                img = pygame.transform.scale(img, (self.square_size, self.square_size))
                self.piece_images[symbol] = img
            except FileNotFoundError:
                print(f"Warning: Missing asset {filename}")

    def reset(self) -> np.ndarray:
        """Reset board to starting position and return observation"""
        self.board.reset()
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute action (move index), return (obs, reward, done, info)"""
        legal_moves = list(self.board.legal_moves)
        if action >= len(legal_moves):
            # Invalid action penalty
            return self._get_observation(), -10.0, False, {"illegal": True}

        move = legal_moves[action]
        captured = self.board.is_capture(move)
        check = self.board.gives_check(move)

        # Execute move
        self.board.push(move)

        # Rewards (custom as per resume: captures, checks, checkmate)
        reward = 0.0
        if captured:
            reward += 2.0  # bonus for capture
        if check:
            reward += 1.5  # bonus for check
        if self.board.is_checkmate():
            reward += 50.0 if self.board.turn == chess.BLACK else -50.0  # agent is white
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.can_claim_draw():
            reward += 0.0  # draw
        else:
            reward -= 0.1  # small living penalty to encourage quick games

        done = self.board.is_game_over()
        truncated = False  # no time limit for now

        info = {
            "move": str(move),
            "captured": captured,
            "check": check,
            "game_over_reason": self.board.result() if done else None
        }

        return self._get_observation(), reward, done, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Simple observation: 8x8x12 one-hot board (12 piece types)"""
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        obs = np.zeros((8, 8, 12), dtype=np.float32)

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                color_offset = 0 if piece.color == chess.WHITE else 6
                type_idx = piece_map[piece.piece_type]
                row, col = divmod(square, 8)
                obs[7 - row, col, color_offset + type_idx] = 1.0  # flip board for white at bottom

        return obs.flatten()  # or keep 8x8x12 if using CNN later

    def render(self):
        if self.render_mode != "human" or not self.screen:
            return

        colors = [(240, 217, 181), (181, 136, 99)]  # light/dark squares
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                pygame.draw.rect(self.screen, color,
                                 (col * self.square_size, row * self.square_size,
                                  self.square_size, self.square_size))

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                symbol = piece.symbol()
                if symbol in self.piece_images:
                    img = self.piece_images[symbol]
                    row, col = divmod(square, 8)
                    self.screen.blit(img, (col * self.square_size, (7 - row) * self.square_size))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None

    def action_space_size(self) -> int:
        """Max possible actions (very large, but we mask invalid in agent)"""
        return 4672  # max legal moves in chess ~218 per position, but we use list len

    def get_legal_moves_mask(self) -> np.ndarray:
        """Binary mask for legal actions (used in agent)"""
        legal_moves = list(self.board.legal_moves)
        mask = np.zeros(self.action_space_size(), dtype=np.float32)
        mask[:len(legal_moves)] = 1.0
        return mask