# import sys
# tag::bot_vs_bot[]
from dlgo import naive
from dlgo import goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move, clear_screen
import time

def main():
    board_size = 9
    game = goboard.GameState.new_game(board_size)
    bots = {
        gotypes.Player.black: naive.RandomBot(),
        gotypes.Player.white: naive.RandomBot(),
    }
    while not game.is_over():
        time.sleep(0.3)  # <1>

        # clear_screen()   # <2>
        print_board(game.board)
        time.sleep(2)
        bot_move = bots[game.next_player].select_move(game)
        print_move(game.next_player, bot_move)
        game = game.apply_move(bot_move)


if __name__ == '__main__':
    main()
