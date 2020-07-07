from dlgo.gotypes import Point

def is_point_an_eye(board, point, color):
    
    # Mot mat la mot diem trong 
    if board.get(point) is not None:
        return False
    
    # Tat ca diem xung quanh mat do phai cung mau 
    for neighbor in point.neighbors():
        if board.is_on_grid(neighbor):
            neighbor_color = board.get(neighbor)
            if neighbor_color != color:
                return False
    
    # Kiem soat 3 tring 4 goc neu diem nam o giua bang;
    # nam tren canh, bat buoc kiem soat tat ca goc 
    friendly_corners = 0
    off_board_corners = 0
    corners = [ 
        Point(point.row - 1, point.col - 1),
        Point(point.row - 1, point.col + 1),
        Point(point.row + 1, point.col - 1),
        Point(point.row + 1, point.col + 1)
    ]
    for corner in corners:
        if board.is_on_grid(corner):
            corner_color = board.get(corner)
            if corner_color == color:
                friendly_corners += 1
        else:
            off_board_corners += 1
    
    if off_board_corners > 0:
        # Diem nam tren canh hoac goc 
        return off_board_corners + friendly_corners == 4
    # Diem nam o giua 
    return friendly_corners >= 3