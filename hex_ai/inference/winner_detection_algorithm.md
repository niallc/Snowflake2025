# Hex Winner Detection Algorithm (Union-Find Approach)

This document describes the winner detection algorithm for Hex, based on the well-tested legacy implementation in `legacy_code/BoardUtils.py`. It outlines the setup, move application, and winner determination using a Union-Find (disjoint-set) data structure.

---

## Overview

The algorithm models the Hex board as a set of pieces, with four special "virtual" pieces representing the board edges. It uses a Union-Find structure to efficiently track connected groups of same-color pieces, including connections to the board edges. A player wins if their two corresponding edge pieces become connected.

---

## 1. Setup: Special Pieces and Connections

- **Special Pieces:**
  - `redLeft`: Represents the left edge for red
  - `redRight`: Represents the right edge for red
  - `blueTop`: Represents the top edge for blue
  - `blueBottom`: Represents the bottom edge for blue
- **Initialization:**
  - Create a new Union-Find structure.
  - Add the four special pieces as disjoint sets.

```python
# Pseudocode
function InitializeConnections(boardWidth):
    connections = new UnionFind()
    specialPieces = {
        redLeft: Piece(0, -1, "red"),
        redRight: Piece(0, boardWidth, "red"),
        blueTop: Piece(-1, 0, "blue"),
        blueBottom: Piece(boardWidth, 0, "blue")
    }
    for piece in specialPieces:
        connections.MakeSet(piece)
    return connections
```

---

## 2. Board Construction

- **For each move in the move list:**
  - Add the piece as a disjoint set to the Union-Find structure.
- **For each move again:**
  - Connect the piece to all same-color neighbors (including special pieces if adjacent) using Union operations.

```python
# Pseudocode
function InitBoardConns(moveList, boardWidth):
    connections = InitializeConnections(boardWidth)
    for move in moveList:
        piece = Piece(move.row, move.col, move.color)
        connections.MakeSet(piece)
    for move in moveList:
        UpdateConnections(connections, move, board)
    return connections
```

---

## 3. Move Application

- **When a move is made:**
  - Find all same-color neighbors (including special pieces if adjacent).
  - Union the new piece with each same-color neighbor.

```python
# Pseudocode
function UpdateConnections(connections, move, board):
    neighbors = getSameColorNeighbors(move, board)
    connections.MakeSet(move)
    for neighbor in neighbors:
        if not connections.hasSet(neighbor):
            connections.MakeSet(neighbor)
        connections.Union(move, neighbor)
```

---

## 4. Winner Detection

- **Red wins** if `redLeft` and `redRight` are connected.
- **Blue wins** if `blueTop` and `blueBottom` are connected.
- Otherwise, there is no winner yet.

```python
# Pseudocode
function FindWinner(connections):
    if connections.AreConnected(redLeft, redRight):
        return "red"
    if connections.AreConnected(blueTop, blueBottom):
        return "blue"
    return "no winner"
```

---

## 5. Notes
- The algorithm is robust to partial board states and can be used to check for a winner at any point in the game.
- Special pieces are treated as part of the board for connection purposes, but are never occupied by a player's move.
- This approach is efficient and well-suited for repeated winner checks during gameplay or analysis.

---

**Reference:**
- See `legacy_code/BoardUtils.py` for the original implementation. 