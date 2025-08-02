var taerim = taerim || {};

taerim.nimbus = {
  LIGHT_BLUE: '#e7fcfc',
  MEDIUM_BLUE: '#bbeeee',
  DARK_BLUE: '#8bd6d6',
  VERY_DARK_BLUE: '#5ab1b1',

  LIGHT_RED: '#fff4ea',
  MEDIUM_RED: '#ffe1c8',
  DARK_RED: '#ffcea5',
  VERY_DARK_RED: '#ffbb82',

  LIGHT_GRAY: '#cccccc',

  // Simply an integer equal to the board size.
  //
  // init() will properly initialize the value to an actual integer; after that,
  // it is constant.
  BOARD_SIZE: null,

  // The name of the coordinate system that should be draw on the board.
  //
  // init() will properly initialize the value to an actual coordinate system
  // name, but it is immutable after initialization.
  COORDINATE_SYSTEM: null,

  // TODO: Find a way to better simulate enums for the difficulty setting and
  // turn flag.

  // Either 'blue' or 'red'.
  NEXT_TURN: 'blue',

  // The value can be 'blue' or 'red' if there is a winner; it must be null if
  // there is not a winner yet.
  WINNER: null,

  // List of moves in order from first to last.
  //
  // Each move is represented by a space string, which is a JSON string for an
  // object with two fields, row and column, representing matrix-0 coordinates
  // with integer values.
  //
  // For example, if the matrix-0 coordinates are (7, 11), 7 being the row index
  // and 11 being the column index, the object would be { row: 7, column: 11 },
  // and the space string would be the result JSON.stringify() on that object.
  MOVES: [],
 
  // Sets of moves that are respectively blue or red. Elements follow the
  // same structure as in the list of moves.
  //
  // The intersection of the two sets must be the empty set. The union must
  // be equal to the list of moves.
  BLUE_MOVES: new Set(),
  RED_MOVES: new Set(),

  // Flag to keep track of whether there is a pending request for a Snowflake
  // operation. While it is true (there is a pending request), the board must be
  // immutable.
  SNOWFLAKE_PENDING: false,

  // Returns a new FormData object that encapsulates the entire state of the
  // game. The FormData can be directly attached to a query to Snowflake.
  makeFormData: function() {
    var formData = new FormData();

    formData.append('difficulty', document.getElementById('difficulty').value);
    formData.append('size', this.BOARD_SIZE);

    var moves = '';
    for (var i = 0; i < this.MOVES.length; i++) {
      moves += this.toTrmphCoordinates(this.MOVES[i]);
    }

    formData.append('moves', moves);

    return formData;
  },

  // Requests a move from Snowflake.
  //
  // If there is already a pending request, does nothing.
  askSnowflake: function() {
    if (this.SNOWFLAKE_PENDING || this.WINNER !== null) {
      return;
    }

    this.SNOWFLAKE_PENDING = true;

    var move_button = document.getElementById('move-button');
    var undo_half_button = document.getElementById('undo-half-button');
    var undo_full_button = document.getElementById('undo-full-button');

    var nimbus = this;

    var xhr = new XMLHttpRequest();
    xhr.timeout = 30000;  // 30 seconds.
    xhr.open('POST', '/snowflake/');
    xhr.ontimeout = function() {
      button.style.backgroundColor = null;
      nimbus.SNOWFLAKE_PENDING = false;
    }
    xhr.onreadystatechange = function() {
      if (xhr.readyState === XMLHttpRequest.DONE) {
        if (xhr.status === 200) {
          var response = JSON.parse(xhr.response);
          var move = null;

          if (response.row !== null && response.column != null) {
            nimbus.applyMove(
                nimbus.makeSpaceString(response.row, response.column));
          }

          if (response.winner === 'blue' || response.winner === 'red') {
            // TODO: Properly differentiate between the game being over without
            // an additional move and the game being over after Snowflake's
            // move.

            nimbus.WINNER = response.winner;
            nimbus.updateStatusLine();
          }
        }

        move_button.style.backgroundColor = null;
        undo_half_button.style.backgroundColor = null;
        undo_full_button.style.backgroundColor = null;
        nimbus.SNOWFLAKE_PENDING = false;
      }
    }

    xhr.send(nimbus.makeFormData());

    move_button.style.backgroundColor = nimbus.LIGHT_GRAY;
    undo_half_button.style.backgroundColor = nimbus.LIGHT_GRAY;
    undo_full_button.style.backgroundColor = nimbus.LIGHT_GRAY;
  },

  // Returns a space string given a row and column in matrix-0 coordinates.
  makeSpaceString: function(row, column) {
    return JSON.stringify({ row: row, column: column }, ['column', 'row']);
  },

  // Returns the DOM element for the box corresponding to the specified space
  // string.
  getBoxElement: function(space) {
    var json = JSON.parse(space);
    return document.getElementById('box-' + json.row + '-' + json.column);
  },

  // Returns the DOM element for the coordindates corresponding to the specified
  // space string.
  getCoordinatesElement: function(space) {
    var json = JSON.parse(space);
    return document.getElementById(
               'coordinates-' + json.row + '-' + json.column);
  },

  // TODO: Consolidate getBoxElement() and getCoordinatesElement() by
  // implementing a getHexElement() function and accessing the hex elements'
  // children.

  // Updates the color of the box corresponding to the specified space string.
  updateBoxColor: function(space) {
    var isLastMove = false;
    if (this.MOVES.length > 0) {
      if (this.MOVES[this.MOVES.length - 1] === space) {
        isLastMove = true;
      }
    }

    var box = this.getBoxElement(space);
    if (this.BLUE_MOVES.has(space)) {
      if (isLastMove) {
        box.style.fill = this.VERY_DARK_BLUE;
      } else {
        box.style.fill = this.DARK_BLUE;
      }
    } else if (this.RED_MOVES.has(space)) {
      if (isLastMove) {
        box.style.fill = this.VERY_DARK_RED;
      } else {
        box.style.fill = this.DARK_RED;
      }
    } else {
      box.style.fill = null;
    }
  },

  // Updates the status line with whose turn it is or who the winner is.
  updateStatusLine: function() {
    var element = document.getElementById('status-line');

    if (this.WINNER === 'blue') {
      element.style.color = this.VERY_DARK_BLUE;
      element.innerText = 'Game over: blue wins';
    } else if (this.WINNER === 'red') {
      element.style.color = this.VERY_DARK_RED;
      element.innerText = 'Game over: red wins';
    } else if (this.NEXT_TURN === 'blue') {
      element.style.color = this.VERY_DARK_BLUE;
      element.innerText = "Blue's turn";
    } else {
      element.style.color = this.VERY_DARK_RED;
      element.innerText = "Red's turn";
    }
  },

  // Flips the color of the next move.
  flipTurn: function() {
		if (this.NEXT_TURN === 'blue') {
			this.NEXT_TURN = 'red';
		} else {
			this.NEXT_TURN = 'blue';
		}

    this.updateStatusLine();
  },

  // Updates the URL to reflect the board state without triggering a refresh.
  updateUrl: function() {
    var path = '/game/' + this.BOARD_SIZE + '/';

    for (var i = 0; i < this.MOVES.length; i++) {
      path += this.toTrmphCoordinates(this.MOVES[i]);
    }

    // if (!path.endsWith('/')) {
    //   path += '/';
    // }

    history.replaceState(null, null, path);
  },

  // Applies a move at the specified space.
  //
  // Returns whether the move is successfully applied.
  //
  // The space must not already be occupied. The resulting color of the space
  // depends simply on whose turn is next.
  //
  // If the move is successfully applied, updates whose turn is next.
  applyMove: function(space) {
    if (this.MOVES.indexOf(space) !== -1) {
      return false;
    }

    var previousMove = null;
    if (this.MOVES.length > 0) {
      previousMove = this.MOVES[this.MOVES.length - 1];
    }

    this.MOVES.push(space);
    if (this.NEXT_TURN === 'blue') {
      this.BLUE_MOVES.add(space);
    } else {
      this.RED_MOVES.add(space);
    }

    this.updateBoxColor(space);
    if (previousMove !== null) {
      this.updateBoxColor(previousMove);
    }

    this.updateUrl();
    this.flipTurn();

    return true;
  },

  // Undoes the specified number of half moves.
  //
  // If the board is in an immutable state (e.g., because there is a pending
  // Snowflake operation) or there are no moves, has no effect.
  undoMove: function(number) {
    if (this.SNOWFLAKE_PENDING) {
      return;
    }

    for (var i = 0; i < number; i++) {
      if (this.MOVES.length === 0) {
        return;
      }

      var space = this.MOVES.pop();
      this.BLUE_MOVES.delete(space);
      this.RED_MOVES.delete(space);

      var previousMove = null;
      if (this.MOVES.length > 0) {
        previousMove = this.MOVES[this.MOVES.length - 1];
      }

      this.updateBoxColor(space);
      if (previousMove !== null) {
        this.updateBoxColor(previousMove);
      }

      this.WINNER = null;
      this.updateUrl();
      this.flipTurn();
    }
  },

  // Either applies a move at the specified matrix-0 coordinates or undoes it.
  //
  // If the board is in an immutable state (e.g., because there is a pending
  // Snowflake operation), has no effect.
  //
  // A move is applied if the clicked box corresponds to an empty space. A move
  // is undone if the clicked box correponds to the space of the most recent
  // move. Otherwise, there is no effect.
  handleBoxClick: function(row, column) {
    if (this.SNOWFLAKE_PENDING || this.WINNER !== null) {
      return;
    }

    var space = this.makeSpaceString(row, column);
    var index = this.MOVES.lastIndexOf(space);

    if (index === -1) {
      // The user clicked an empty box. Apply the move.
      this.applyMove(space);

      var automatic_checkbox = document.getElementById('automatic');
      if (automatic_checkbox.checked) {
        this.askSnowflake();
      }
    } else {
      // The user clicked a non-empty box. Was it the most recent move?
      if (index === this.MOVES.length - 1) {
        this.undoMove();
      }
    }
  },

  // Converts a integral matrix-0 coordinate to a character coordinate suitable
  // for SGF or TRMPH coordinates.
  //
  // For example, the matrix-0 coordinate 3 would be converted to the character
  // coordinate 'c'.
  toCharCoordinate: function(n) {
    return String.fromCharCode(n + 97);
  },

  // Converts a space string to a string for the TRMPH coordinates of the same
  // space.
  toTrmphCoordinates: function(space) {
    var json = JSON.parse(space);
    return this.toCharCoordinate(json.column) + (json.row + 1);
  },

  // Draws coordinates on each space on the board according to the specified
  // coordinate system, which must be one of 'Screen', 'SGF', 'TRMPH', or
  // 'None'.
  drawCoordinates: function(system) {
    for (var r = 0; r < this.BOARD_SIZE; r++) {
      for (var c = 0; c < this.BOARD_SIZE; c++) {
        var coordinates = null;
        if (system === 'Matrix-0') {
          coordinates = r + ',' + c;
        } else if (system === 'Matrix-1') {
          coordinates = (r + 1) + ',' + (c + 1);
        } else if (system === 'Screen') {
          coordinates = c + ',' + r;
        } else if (system === 'SGF') {
          coordinates =
              '[' + this.toCharCoordinate(c) + this.toCharCoordinate(r) + ']';
        } else if (system === 'TRMPH') {
          coordinates = this.toCharCoordinate(c) + (r + 1);
        }

        this.getCoordinatesElement(this.makeSpaceString(r, c)).textContent =
            coordinates;
      }
    }

    this.COORDINATE_SYSTEM = system;
  },

  // Restarts the game by triggering a page refresh and directing the user to an
  // empty board of the desired size.
  restart: function() {
    window.location.href =
        '/game/' + document.getElementById('selected-size').value + '/';

    var board_size = document.getElementById('selected-size').value;
    var target = '/game/' + board_size + '/';

    var difficulty = document.getElementById('difficulty').value;
    target += '?d=' + difficulty;

    var automatic = document.getElementById('automatic').checked;
    target += '&a=' + automatic;

    var coordinate_system = this.COORDINATE_SYSTEM;
    target += '&c=' + coordinate_system;

    window.location.href = target;
  },

  // Initializes JavaScript state and sets up the page.
  init: function(size, coordinate_system, moves = []) {
    this.BOARD_SIZE = size;
    this.updateUrl();
    this.drawCoordinates(coordinate_system);

    for (var i = 0; i < moves.length; i++) {
      if (!this.applyMove(JSON.stringify(moves[i], ['column', 'row']))) {
        throw null;
      }
    }

    this.updateStatusLine();
  },
};
