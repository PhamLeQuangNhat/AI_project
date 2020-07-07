import copy
from dlgo.gotypes import Player, Point
from dlgo.scoring import compute_game_result
import dlgo.zobrist as zobrist

__all__ = [
    'Board',
    'GameState',
    'Move',
]

neighbor_tables = {}
corner_tables = {}


def init_neighbor_table(dim):
    rows, cols = dim
    new_table = {}
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            p = Point(row=r, col=c)
            full_neighbors = p.neighbors()
            true_neighbors = [
                n for n in full_neighbors
                if 1 <= n.row <= rows and 1 <= n.col <= cols]
            new_table[p] = true_neighbors
    neighbor_tables[dim] = new_table


def init_corner_table(dim):
    rows, cols = dim
    new_table = {}
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            p = Point(row=r, col=c)
            full_corners = [
                Point(row=p.row - 1, col=p.col - 1),
                Point(row=p.row - 1, col=p.col + 1),
                Point(row=p.row + 1, col=p.col - 1),
                Point(row=p.row + 1, col=p.col + 1),
            ]
            true_corners = [
                n for n in full_corners
                if 1 <= n.row <= rows and 1 <= n.col <= cols]
            new_table[p] = true_corners
    corner_tables[dim] = new_table


class IllegalMoveError(Exception):
    pass

# class Move gom 3 hanh dong (play, pass, resign)
class Move():

    # bat ki hanh dong nao cua nguoi choi co the choi 
    # thong qua is_play, is_pass hoac is_resign 
    def __init__(self, point=None, is_pass=False, is_resign=False):
        assert (point is not None) ^ is_pass ^ is_resign
        self.point = point
        self.is_play = (self.point is not None)
        self.is_pass = is_pass
        self.is_resign = is_resign

    @classmethod
    # Dat mot vien da len ban co
    def play(cls,point):
        return Move(point=point)
    
    @classmethod
    # Nuoc di bi bo qua
    def pass_turn(cls):
        return Move(is_pass=True)

    @classmethod
    # Nuoc di bo cuoc 
    def resign(cls):
        return Move(is_resign=True)

#  Da duoc ket noi boi mot chuoi cac vien da cung mau 
class GoString():
    def __init__(self, color, stones, liberties):
        self.color = color
        self.stones = frozenset(stones)
        self.liberties = frozenset(liberties)
    
    # Xoa o tu do 
    def without_liberty(self, point):
        new_liberties = self.liberties - set([point])
        return GoString(self.color, self.stones, new_liberties)

    # Them o tu do 
    def with_liberty(self, point):
        new_liberties = self.liberties | set([point])
        return GoString(self.color, self.stones, new_liberties)

    # Tra ve mot chuoi moi voi tat ca cac vien da trong 2 chuoi
    def merged_with(self, string):
        assert string.color == self.color
        combined_stones = self.stones | string.stones
        return GoString(self.color,combined_stones,
            (self.liberties | string.liberties) - combined_stones)

    @property
    # So luong cac o tu do tai bat ki diem trong bang
    def num_liberties(self):
        return len(self.liberties)

    def __eq__(self, other):
        return isinstance(other, GoString) and \
            self.color == other.color and \
            self.stones == other.stones and \
            self.liberties == other.liberties

    def __deepcopy__(self, memodict={}):
        return GoString(self.color, self.stones, copy.deepcopy(self.liberties))

# class Board 
class Board():
    # khoi tao bang 
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        # luu tru cac chuoi da 
        self._grid = {}
        self._hash = zobrist.EMPTY_BOARD

        global neighbor_tables
        dim = (num_rows, num_cols)
        if dim not in neighbor_tables:
            init_neighbor_table(dim)
        if dim not in corner_tables:
            init_corner_table(dim)
        self.neighbor_table = neighbor_tables[dim]
        self.corner_table = corner_tables[dim]

    def neighbors(self, point):
        return self.neighbor_table[point]

    def corners(self, point):
        return self.corner_table[point]

    # kiem tra tat ca cac vien da lan can cua 1 diem co tu do hay khong
    def place_stone(self, player, point):
        assert self.is_on_grid(point)
        if self._grid.get(point) is not None:
            print('Illegal play on %s' % str(point))
        assert self._grid.get(point) is None
        # Kiem tra cac diem lien ke 
        adjacent_same_color = []
        adjacent_opposite_color = []
        liberties = []
        
        # kiem tra cac diem ho hang cua diem dang xet  
        for neighbor in self.neighbor_table[point]:
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                liberties.append(neighbor)
            elif neighbor_string.color == player:
                if neighbor_string not in adjacent_same_color:
                    adjacent_same_color.append(neighbor_string)
            else:
                if neighbor_string not in adjacent_opposite_color:
                    adjacent_opposite_color.append(neighbor_string)
        new_string = GoString(player, [point], liberties)

        # Hop nhat bat ki chuoi lien ke co cung mau 
        for same_color_string in adjacent_same_color:
            new_string = new_string.merged_with(same_color_string)
        for new_string_point in new_string.stones:
            self._grid[new_string_point] = new_string
        # Xoa mot diem trong = hash code 
        self._hash ^= zobrist.HASH_CODE[point, None]
        # Them mot diem = hash code 
        self._hash ^= zobrist.HASH_CODE[point, player]

        for other_color_string in adjacent_opposite_color:
            replacement = other_color_string.without_liberty(point)
            # Giam o tu do cua bat ki chuoi mau doi lap lien ke 
            if replacement.num_liberties:
                self._replace_string(other_color_string.without_liberty(point))
            # Neu bat ki chuoi doi lap khong co o tu do, xoa chuoi 
            else:
                self._remove_string(other_color_string)

    # thay the mot chuoi 
    def _replace_string(self, new_string):
        for point in new_string.stones:
            self._grid[point] = new_string

    # loai bo mot chuoi 
    def _remove_string(self, string):
        for point in string.stones:
            # Loai bo mot chuoi co the tang so luong o tu do cho chuoi khac
            for neighbor in self.neighbor_table[point]:
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None:
                    continue
                if neighbor_string is not string:
                    self._replace_string(neighbor_string.with_liberty(point))
            self._grid[point] = None
            # Xoa mot diem trong = hash code 
            self._hash ^= zobrist.HASH_CODE[point, string.color]
            # Them mot diem trong = hash code 
            self._hash ^= zobrist.HASH_CODE[point, None]

    def is_self_capture(self, player, point):
        friendly_strings = []
        for neighbor in self.neighbor_table[point]:
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                # Diem nay co mot o tu do, khong the la self capture 
                return False
            elif neighbor_string.color == player:
                # Tap hop de phan tich sau 
                friendly_strings.append(neighbor_string)
            else:
                if neighbor_string.num_liberties == 1:
                    # Day la nuoc di thuc su bat quan, khong the la self capture.
                    return False
        if all(neighbor.num_liberties == 1 for neighbor in friendly_strings):
            return True
        return False

    def will_capture(self, player, point):
        for neighbor in self.neighbor_table[point]:
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                continue
            elif neighbor_string.color == player:
                continue
            else:
                if neighbor_string.num_liberties == 1:
                    # Day la nuoc di se bat quan
                    return True
        return False

    # kiem tra mot diem co nam trong gioi han bang hay khong 
    def is_on_grid(self, point):
        return 1 <= point.row <= self.num_rows and \
            1 <= point.col <= self.num_cols

    # Tra ve noi dung cua mot diem tren bang. Tra ve None neu diem trong 
    # hoac tra mau sac cua da neu co vien da o diem do 
    def get(self, point):
        string = self._grid.get(point)
        if string is None:
            return None
        return string.color

    # Tra ve toan bo chuoi da tai 1 diem. Tra ve None neu diem trong 
    # hoac tra ve mot chuoi neu da co ton tai o diem do 
    def get_go_string(self, point):
        string = self._grid.get(point)
        if string is None:
            return None
        return string

    def __eq__(self, other):
        return isinstance(other, Board) and \
            self.num_rows == other.num_rows and \
            self.num_cols == other.num_cols and \
            self._hash() == other._hash()

    def __deepcopy__(self, memodict={}):
        copied = Board(self.num_rows, self.num_cols)
        copied._grid = copy.copy(self._grid)
        copied._hash = self._hash
        return copied

    def zobrist_hash(self):
        return self._hash

# class GameState cho thong tin ve vi tri bang, nguoi choi ke tiep,
# trang thai gam truoc va nuoc di cuoi cung da duoc choi 
class GameState():
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        if previous is None:
            self.previous_states = frozenset()
        else:
            self.previous_states = frozenset(
                previous.previous_states |
                {(previous.next_player, previous.board.zobrist_hash())})
        self.last_move = move

    # Tra ve trang thai moi cua GameState sau khi di mot nuoc di 
    def apply_move(self, move):
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board
        return GameState(next_board, self.next_player.other, self, move)

    # Khi tro choi bat dau 
    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        return GameState(board, Player.black, None, None)

    # Khi tro choi ket thuc
    def is_over(self):
        if self.last_move is None:
            return False
        if self.last_move.is_resign:
            return True
        second_last_move = self.previous_state.last_move
        if second_last_move is None:
            return False
        return self.last_move.is_pass and second_last_move.is_pass

    # Kiem tra xem co viec bat giu co hop le hay khong 
    def is_move_self_capture(self, player, move):
        if not move.is_play:
            return False
        return self.board.is_self_capture(player, move.point)
    # Qui tac Ko 
    @property
    def situation(self):
        return (self.next_player, self.board)

    def does_move_violate_ko(self, player, move):
        if not move.is_play:
            return False
        if not self.board.will_capture(player, move.point):
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        next_situation = (player.other, next_board.zobrist_hash())
        return next_situation in self.previous_states

    # Kiem tra nuoc di hop le 
    def is_valid_move(self, move):
        if self.is_over():
            return False
        if move.is_pass or move.is_resign:
            return True
        return (
            self.board.get(move.point) is None and
            not self.is_move_self_capture(self.next_player, move) and
            not self.does_move_violate_ko(self.next_player, move))

    # Nhung nuoc di hop phap 
    def legal_moves(self):
        if self.is_over():
            return []
        moves = []
        for row in range(1, self.board.num_rows + 1):
            for col in range(1, self.board.num_cols + 1):
                move = Move.play(Point(row, col))
                if self.is_valid_move(move):
                    moves.append(move)
        # Day la 2 nuoc di luon luon hop phap
        moves.append(Move.pass_turn())
        moves.append(Move.resign())
        return moves

    # Nguoi chien thang 
    def winner(self):
        if not self.is_over():
            return None
        if self.last_move.is_resign:
            return self.next_player
        game_result = compute_game_result(self)
        return game_result.winner