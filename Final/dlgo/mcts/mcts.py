import random, math
from dlgo.gotypes import Player
from dlgo import agent
from dlgo.utils import coords_from_point


__all__ = [
    'MCTSAgent',
]

class MCTSNode(object):
    def __init__(self, game_state, parent=None, move=None):
        # Trang thai hien tai cua tro choi 
        self.game_state = game_state 
        self.parent = parent
        self.move = move 

        # Thong ke ve viec trien khai bat dau tu node nay 
        self.win_counts = {
            Player.black: 0,
            Player.white: 0,
        }
        self.num_rollouts = 0

        # Danh sach tat ca cac node con 
        self.children = []

        # Danh sach tat ca cac dong thai hop phap chua duoc them vao cay 
        self.unvisited_moves = game_state.legal_moves()

    def add_random_child(self):
        # Chon ngau nhien mỏ node la moi va them no vao cay 
        index = random.randint(0, len(self.unvisited_moves) - 1)
        new_move = self.unvisited_moves.pop(index)
        new_game_state = self.game_state.apply_move(new_move)
        new_node = MCTSNode(new_game_state, self, new_move)
        self.children.append(new_node)
        return new_node
    
    def record_win(self, winner):
        # Cap nhat so lieu thong ke 
        self.win_counts[winner] += 1
        self.num_rollouts += 1
    
    def can_add_child(self):
        # Tra ve so luong nuoc di hop phap chua duoc them vao cay 
        return len(self.unvisited_moves) > 0
    
    def is_terminal(self):
        # Game ket thuc 
        return self.game_state.is_over()
    
    def winning_pct(self, player):
        # Tra ve ty le phan tram thang cua mot nguoi choi cu the 
        return float(self.win_counts[player]) / float(self.num_rollouts)
    
class MCTSAgent(agent.Agent):
    def __init__(self, num_rounds, temperature):
        agent.Agent.__init__(self)
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_move(self, game_state):
        # Chon nhanh tot nhat cua MCT 

        # Tao mot cay moi bat dau tu trang thai hien tai 
        root = MCTSNode(game_state)

        # Tao mot cay voi so chieu rong co dinh (so lan co dinh chap nhan duoc)
        for i in range(self.num_rounds):
            node = root

            # Tim mot node co di chuyen hop le va chua ket thuc tro choi 
            while (not node.can_add_child()) and (not node.is_terminal()):

                # Node duoc tim kiem tiep theo duoc chon theo diem UCT 
                node = self.select_child(node)

            #  1 nut moi duoc them ngau nhien vao node 
            if node.can_add_child():
                node = node.add_random_child()

            # Huong dan nguoi choi chien thang khi di chuyen 
            winner = self.simulate_random_game(node.game_state)

            # Theo doi cay va ghi lai diem so 
            while node is not None:
                node.record_win(winner)
                node = node.parent

        scored_moves = [
            (child.winning_pct(game_state.next_player), child.move, child.num_rollouts)
            for child in root.children
        ]
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        for s, m, n in scored_moves[:10]:
            print('%s - %.3f (%d)' % (m, s, n))

        # Chon va tra ve huong di co ti le thang cao nhat tu mo phong 
        best_move = None
        best_pct = -1.0
        for child in root.children:
            child_pct = child.winning_pct(game_state.next_player)
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move
        print('Select move %s with win pct %.3f' % (best_move, best_pct))
        return best_move
    
    # Tra ve nut co diem UCT cao nhat , duoc su dung de chon node tiep theo 
    def select_child(self, node):
        
        total_rollouts = sum(child.num_rollouts for child in node.children)
        best_score = -1
        best_child = None
        for child in node.children:
            score = self.uct_score(
                total_rollouts,
                child.num_rollouts,
                child.winning_pct(node.game_state.next_player),
                self.temperature
            )
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    @staticmethod
    def uct_score(parent_rollouts, child_rollouts, win_pct, temperature):
        exploration = math.sqrt(math.log(parent_rollouts) / child_rollouts)
        return win_pct + temperature * exploration

    @staticmethod
    def simulate_random_game(game):
        # Bat dau trien khai tu node nay
        bots = {
            Player.black: agent.RandomBot(),
            Player.white: agent.RandomBot(),
        }

        while not game.is_over():
            bot_move = bots[game.next_player].select_move(game)
            game = game.apply_move(bot_move)
        
        return game.winner()