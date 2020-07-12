import os

from flask import Flask
from flask import jsonify
from flask import request

from dlgo import agent
from dlgo import goboard
from dlgo.utils import coords_from_point
from dlgo.utils import point_from_coords
#from tensorflow import get_default_graph
import tensorflow as tf
__all__ = [
    'get_web_app',
]

#graph = get_default_graph()
graph = tf.compat.v1.get_default_graph()
def get_web_app(bot_map):
    
    here = os.path.dirname(__file__)
    static_path = os.path.join(here, 'static')
    app = Flask(__name__, static_folder=static_path, static_url_path='/static')

    @app.route('/select-move/<bot_name>', methods=['POST'])
    def select_move(bot_name):
        global graph

        content = request.json
        board_size = content['board_size']
        game_state = goboard.GameState.new_game(board_size)
        # Replay the game up to this point.
        for move in content['moves']:
            if move == 'pass':
                next_move = goboard.Move.pass_turn()
            elif move == 'resign':
                next_move = goboard.Move.resign()
            else:
                next_move = goboard.Move.play(point_from_coords(move))
            game_state = game_state.apply_move(next_move)
        bot_agent = bot_map[bot_name]
    
        with graph.as_default():
            bot_move = bot_agent.select_move(game_state)
    
        if bot_move.is_pass:
            bot_move_str = 'pass'
        elif bot_move.is_resign:
            bot_move_str = 'resign'
        else:
            bot_move_str = coords_from_point(bot_move.point)
        return jsonify({
            'bot_move': bot_move_str,
            # 'diagnostics': bot_agent.diagnostics()
        })

    return app
