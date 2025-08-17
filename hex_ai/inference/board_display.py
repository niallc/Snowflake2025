import numpy as np
from hex_ai.utils.format_conversion import board_2nxn_to_nxn
from hex_ai.enums import Piece, char_to_piece, get_piece_unicode_symbol
import sys

def ansi_colored(text, color):
    colors = {
        'blue': '\033[34m',
        'red': '\033[31m',
        'reset': '\033[0m',
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"

def display_hex_board(board: np.ndarray, file=None, highlight_move=None) -> None:
    """
    Display a Hex board (NxN format) as ASCII art, with optional move highlighting.
    Args:
        board: np.ndarray of shape (N, N) or (2, N, N), values 'e'=empty, 'b'=blue, 'r'=red or 2-channel format
        file: file-like object to write to (default: stdout)
        highlight_move: (row, col) tuple to highlight, or None
    """
    # TODO: ENUM MIGRATION - Upstream callers should pass domain types consistently.
    # Convert (2, N, N) format to (N, N) if needed
    if board.ndim == 3 and board.shape[0] == 2:
        board = board_2nxn_to_nxn(board)
    N = board.shape[0]
    
    highlight_symbol = '*'  # Symbol for highlighted move
    lines = []
    header = '   ' + ' '.join(str(i) for i in range(N))
    lines.append(header)
    use_color = file is None and sys.stdout.isatty()
    
    for row in range(N):
        indent = ' ' * row
        row_str = f"{row:2d} " + indent
        for col in range(N):
            piece_char = board[row, col]
            piece = char_to_piece(piece_char)
            symbol = get_piece_unicode_symbol(piece)
            
            if highlight_move is not None and (row, col) == highlight_move:
                row_str += highlight_symbol + ' '
            else:
                if use_color:
                    if piece == Piece.BLUE:
                        symbol = ansi_colored(symbol, 'blue')
                    elif piece == Piece.RED:
                        symbol = ansi_colored(symbol, 'red')
                row_str += symbol + ' '
        lines.append(row_str)
    
    output = '\n'.join(lines)
    if file is not None:
        print(output, file=file)
    else:
        print(output) 