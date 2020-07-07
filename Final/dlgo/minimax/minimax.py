import enum
import random

from dlgo.agent import Agent

__all__ = [
    'MinimaxAgent',
]

class GameResult(enum.Enum):
    loss = 1
    draw = 2
    win = 3

def reverse_game_result(game_result):
    if game_result == GameResult.loss:
        return game_result.win
    if game_result == GameResult.win:
        return game_result.loss
    return GameResult.draw

def best_result(game_state):
    if game_state.is_over():
        # Neu tro choi ket thuc
        if game_state.winner() == game_state.next_player:
            # Minh thang 
            return GameResult.win
        elif game_state.winner() is None:
            # Hoa
            return GameResult.draw
        else:
            # Doi thu thang
            return GameResult.loss

    best_result_so_far = GameResult.loss
    for candidate_move in game_state.legal_moves():
        # Trang thai tiep theo khi di 1 nuoc
        next_state = game_state.apply_move(candidate_move)
        # Tim nuoc di tot nhat cho doi thu 
        opponent_best_result = best_result(next_state)         
        # Ket qua ccua minh se trai nguoc ket qua cua doi thu 
        our_result = reverse_game_result(opponent_best_result) 
        # So sanh ket qua tai voi ket qua tot truoc day 
        # de tim ket qua tot nhat 
        if our_result.value > best_result_so_far.value:       
            best_result_so_far = our_result
    return best_result_so_far

class MinimaxAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
    def select_move(self, game_state):
        winning_moves = []
        draw_moves = []
        losing_moves = []
        # Lap qua tat ca cac nuoc di hop le 
        for possible_move in game_state.legal_moves():
            # Tinh toan trang thai neu ta chon 1 nuoc di 
            next_state = game_state.apply_move(possible_move)
            # Vi doi thu choi luot tiep theo, tim ra ket qua 
            # tot nhat co the tu nuoc di hien tai
            opponent_best_outcome = best_result(next_state)
            # Ket qua cua minh se trai nguoc ket qua cua doi thu 
            our_best_outcome = reverse_game_result(opponent_best_outcome)
            # Them di chuyen nay vao danh sach nuoc di 
            if our_best_outcome == GameResult.win:
                winning_moves.append(possible_move)
            elif our_best_outcome == GameResult.draw:
                draw_moves.append(possible_move)
            else:
                losing_moves.append(possible_move)
        if winning_moves:
            return random.choice(winning_moves)
        if draw_moves:
            return random.choice(draw_moves)
        return random.choice(losing_moves)
