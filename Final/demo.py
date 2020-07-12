from six.moves import input
import time 
from dlgo import goboard
from dlgo import gotypes
from dlgo import minimax
from dlgo.utils import print_board, print_move, point_from_coords
from dlgo import mcts
from dlgo.httpfrontend.server import get_web_app

def capture_diff(game_state):
    black_stones = 0
    white_stones = 0
    for r in range(1, game_state.board.num_rows + 1):
        for c in range(1, game_state.board.num_cols + 1):
            p = gotypes.Point(r, c)
            color = game_state.board.get(p)
            if color == gotypes.Player.black:
                black_stones += 1
            elif color == gotypes.Player.white:
                white_stones += 1
    diff = black_stones - white_stones                    
    if game_state.next_player == gotypes.Player.black:    
        return diff                                       
    return -1 * diff   

def main():
    print("******************************************************************")
    print("*                                                                *")
    print("*    <3 <3 <3 <3 <3     WELCOME TO GAME GO     <3 <3 <3 <3 <3    *")
    print("*                                                                *")
    print("******************************************************************")
    print("*                                                                *")
    print("*         1. Play game on terminal                               *")
    print("*                                                                *")
    print("*             a. Human vs Bot AlphaBeta on Board 9x9             *") 
    print("*             b. Human vs Bot Depthprune on Board 9x9            *")
    print("*             c. Human vs Bot MCTS on Board 9x9                  *")
    print("*             d. Bot AlphaBeta vs Bot MCTS on Board 9x9          *")
    print("*                                                                *")
    print("*         2. Play game on web                                    *")
    print("*                                                                *")
    print("*             a. Human vs Bot MCTS on Board 9x9                  *")
    print("*             b. Human vs Bot DeepLearning on Board 19x19        *")
    print("*                                                                *")
    print("******************************************************************")
    print("                                                                  ")
    print("            *****************************************             ")
    print("                                                                  ")
    choices_A = int(input("                     Choose Terminal or Web: "))
    choices_B =     input("                         Choose type bot: ")
    print("                                                                  ")
    print("            *****************************************             ")
    BOARD_SIZE = 9
    game = goboard.GameState.new_game(BOARD_SIZE)

    if choices_A == 1:
        if choices_B == 'a':
            bot = minimax.AlphaBetaAgent(4, capture_diff)
        if choices_B == 'b':
            bot = minimax.DepthPrunedAgent(4, capture_diff)
        if choices_B == 'c':
            bot = mcts.MCTSAgent(500, temperature=1.4)
        if choices_B == 'd':
            bots = {
                gotypes.Player.black: minimax.AlphaBetaAgent(4, capture_diff),
                gotypes.Player.white: mcts.MCTSAgent(500, temperature=1.4),
            }
            while not game.is_over():
                time.sleep(0.3)    
                print_board(game.board)
                bot_move = bots[game.next_player].select_move(game)
                print_move(game.next_player, bot_move)
                game = game.apply_move(bot_move)

        if choices_B == 'a' or choices_B == 'b' or choices_B == 'c':
            while not game.is_over():
                print_board(game.board)
                if game.next_player == gotypes.Player.black:
                    human_move = input('-- ')
                    point = point_from_coords(human_move.strip())
                    move = goboard.Move.play(point)
                else:
                    move = bot.select_move(game)
                print_move(game.next_player, move)
                game = game.apply_move(move)
    else:
        if choices_B == 'a':
            bot = mcts.MCTSAgent(700, temperature=1.4)
            web_app = get_web_app({'mcts':bot})
            web_app.run()

if __name__ == '__main__':
    main()