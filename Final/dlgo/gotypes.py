from collections import namedtuple 
# su dung enum de bieu thi mau sac khac nhau cua da 
import enum 

# sau khi nguoi choi dat da, co the chuyen mau bang 
# goi phung phap "other" trong class Player 
class Player(enum.Enum):
    black = 1
    white = 2

    @property 
    def other(self):
        if self == Player.white:
            return Player.black
        else:
            return Player.white
    
# toa do bang 
class Point(namedtuple('Point','row col')):
    def neighbors(self):
        return [Point(self.row - 1, self.col),
                Point(self.row + 1, self.col),
                Point(self.row, self.col - 1),
                Point(self.row, self.col + 1),
                ]

    