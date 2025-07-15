import numpy as np

def display_hex_board(board: np.ndarray, file=None):
    """
    Display a Hex board (NxN trinary format) as ASCII art.
    Args:
        board: np.ndarray of shape (N, N), values 0=empty, 1=blue, 2=red
        file: file-like object to write to (default: stdout)
    """
    N = board.shape[0]
    symbols = {0: '.', 1: 'B', 2: 'R'}
    lines = []
    header = '   ' + ' '.join(str(i) for i in range(N))
    lines.append(header)
    for row in range(N):
        indent = '  ' * row
        row_str = f"{row:2d} " + indent + ' '.join(symbols[board[row, col]] for col in range(N))
        lines.append(row_str)
    output = '\n'.join(lines)
    if file is not None:
        print(output, file=file)
    else:
        print(output) 