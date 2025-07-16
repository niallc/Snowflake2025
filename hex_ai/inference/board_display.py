import numpy as np
from hex_ai.utils.format_conversion import board_2nxn_to_nxn

def display_hex_board(board: np.ndarray, file=None, highlight_move=None):
    """
    Display a Hex board (NxN trinary format) as ASCII art, with optional move highlighting.
    Args:
        board: np.ndarray of shape (N, N) or (2, N, N), values 0=empty, 1=blue, 2=red or 2-channel format
        file: file-like object to write to (default: stdout)
        highlight_move: (row, col) tuple to highlight, or None
    """
    # Convert (2, N, N) format to (N, N) if needed
    if board.ndim == 3 and board.shape[0] == 2:
        board = board_2nxn_to_nxn(board)
    N = board.shape[0]
    symbols = {0: '.', 1: 'B', 2: 'R'}
    highlight_symbol = '*'  # Symbol for highlighted move
    lines = []
    header = '   ' + ' '.join(str(i) for i in range(N))
    lines.append(header)
    for row in range(N):
        indent = '  ' * row
        row_str = f"{row:2d} " + indent
        for col in range(N):
            if highlight_move is not None and (row, col) == highlight_move:
                row_str += highlight_symbol + ' '
            else:
                row_str += symbols[board[row, col]] + ' '
        lines.append(row_str)
    output = '\n'.join(lines)
    if file is not None:
        print(output, file=file)
    else:
        print(output) 