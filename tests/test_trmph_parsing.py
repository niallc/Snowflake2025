import unittest
import numpy as np
from hex_ai.data_utils import (
    strip_trmph_preamble, split_trmph_moves, trmph_move_to_rowcol, parse_trmph_to_board
)
from hex_ai.config import BOARD_SIZE

class TestTrmphParsing(unittest.TestCase):
    def test_strip_trmph_preamble(self):
        s = "http://www.trmph.com/hex/board#13,a1b2c3"
        self.assertEqual(strip_trmph_preamble(s), "a1b2c3")
        s2 = "#13,a1b2c3"
        self.assertEqual(strip_trmph_preamble(s2), "a1b2c3")
        with self.assertRaises(ValueError):
            strip_trmph_preamble("a1b2c3")

    def test_split_trmph_moves(self):
        self.assertEqual(split_trmph_moves("a1b2c3"), ["a1", "b2", "c3"])
        self.assertEqual(split_trmph_moves("a1m13"), ["a1", "m13"])
        with self.assertRaises(ValueError):
            split_trmph_moves("1a2b")

    def test_trmph_move_to_rowcol(self):
        self.assertEqual(trmph_move_to_rowcol("a1"), (0, 0))
        self.assertEqual(trmph_move_to_rowcol("m13"), (12, 12))
        with self.assertRaises(ValueError):
            trmph_move_to_rowcol("z1")
        with self.assertRaises(ValueError):
            trmph_move_to_rowcol("a14")

    def test_parse_trmph_to_board(self):
        s = "#13,a1b2c3"
        board = parse_trmph_to_board(s)
        expected = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        expected[0, 0] = 1  # a1, blue
        expected[1, 1] = 2  # b2, red
        expected[2, 2] = 1  # c3, blue
        np.testing.assert_array_equal(board, expected)
        # Test duplicate move
        s_dup = "#13,a1a1"
        with self.assertRaises(ValueError):
            parse_trmph_to_board(s_dup)

if __name__ == "__main__":
    unittest.main() 