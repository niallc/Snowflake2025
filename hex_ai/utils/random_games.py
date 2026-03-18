import numpy as np

import hex_ai.utils.format_conversion as fc

def generate_random_board_rowcol(board_size: int, num_moves: int) -> list:
    move_list = []
    i = 0
    while i < num_moves:
        row = np.random.randint(0, board_size)
        col = np.random.randint(0, board_size)
        if (row, col) not in move_list:
            move_list.append((row, col))
            i += 1
    return move_list

def convert_move_list_rowcol_to_trmph(move_list: list) -> str:
    trmph_str = ""
    for row, col in move_list:
        trmph_str += fc.rowcol_to_trmph(row, col)
    return trmph_str

def generate_random_board_trmph(board_size: int, num_moves: int) -> str:
    move_list = generate_random_board_rowcol(board_size, num_moves)
    return convert_move_list_rowcol_to_trmph(move_list)