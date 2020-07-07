import random

from dlgo.agent import Agent
from dlgo.scoring import GameResult

__all__ = [
    'DepthPrunedAgent',
]

MAX_SCORE = 999999
MIN_SCORE = -999999


def reverse_game_result(game_result):
    if game_result == GameResult.loss:
        return game_result.win
    if game_result == GameResult.win:
        return game_result.loss
    return GameResult.draw

def best_result(game_state, max_depth, eval_fn):
    # Kiem tra xem tro choi ket thuc chua 
    if game_state.is_over():                               
        if game_state.winner() == game_state.next_player: 
            return MAX_SCORE                               
        else:                                              
            return MIN_SCORE                               
    
    # Khi dat den do sau tim kiem toi da, 
    # su dung ham danh gia de quyet dinh nuoc di tot nhat 
    if max_depth == 0:                                     
        return eval_fn(game_state)                         

    best_so_far = MIN_SCORE
    # Lap qua tat ca cac nuoc di hop le 
    for candidate_move in game_state.legal_moves():        
        # Trang thai tiep theo khi di 1 nuoc
        next_state = game_state.apply_move(candidate_move)
        # Tim nuoc di tot nhat cho doi thu tu vi tri hien tai
        opponent_best_result = best_result(                
            next_state, max_depth - 1, eval_fn)            
        # Ket qua cua minh se trai nguoc ket qua cua doi thu 
        our_result = -1 * opponent_best_result             
        # Xem day co phai nuoc di tot nhat tu truoc toi nay 
        if our_result > best_so_far:                       
            best_so_far = our_result                       

    return best_so_far

class DepthPrunedAgent(Agent):
    def __init__(self, max_depth, eval_fn):
        Agent.__init__(self)
        self.max_depth = max_depth
        self.eval_fn = eval_fn

    def select_move(self, game_state):
        best_moves = []
        best_score = None
        # Lap qua tat ca cac nuoc di hop le 
        for possible_move in game_state.legal_moves():
            # Tinh toan trang thai neu ta chon 1 nuoc di 
            next_state = game_state.apply_move(possible_move)
            # Vi doi thu choi luot tiep theo, tim ra ket qua 
            # tot nhat co the tu nuoc di hien tai
            opponent_best_outcome = best_result(next_state, self.max_depth, self.eval_fn)
            # Ket qua cua minh se trai nguoc ket qua cua doi thu 
            our_best_outcome = -1 * opponent_best_outcome
            if (not best_moves) or our_best_outcome > best_score:
                # Day la nuoc di tot nhat tu truoc toi nay 
                best_moves = [possible_move]
                best_score = our_best_outcome
            elif our_best_outcome == best_score:
                # Them di chuyen nay vao danh sach nuoc di 
                best_moves.append(possible_move)
        # Chon ngau nhien trong so tat ca cac di chuyen tot nhu nhau 
        return random.choice(best_moves)
