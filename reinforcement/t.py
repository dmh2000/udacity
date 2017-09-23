import numpy as np

wins = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6]
]

def printBoard(board):
    global count

    count += 1
    k = 0
    for i in range(3):
        for j in range(3):
            print(board[k]),
            k += 1
        print('\n')
    print('====================')

def winX(board):
    global wins

    # check x's
    for w in wins:
        win = board[w[0]] == 'X' and board[w[1]] == 'X' and board[w[2]] == 'X'
        if win:
            return 'X'
    return False


def winO(board):
    global wins

    # check x's
    win = False
    for w in wins:
        win = board[w[0]] == 'O' and board[w[1]] == 'O' and board[w[2]] == 'O'
        if win:
            return 'O'
    return False

def score(board):
    if winX(board):
        return 10
    elif winO(board):
        return -10
    else:
        return 0

def O(board):
    s = score(board)
    if s == -10:
        printBoard(board)
        return s

    # get list of empty cells in the board (moves)
    moves = []
    for i in range(len(board)):
        # if cell is empty
        if board[i] == '-':
            moves.append(i)

    scores = []
    for m in moves:
        board[m] = 'O'
        s = X(board)
        scores.append(s)
        board[m] = '-'

    if len(scores) > 0:
        s = np.min(scores)
    else:
        s = 0
    return s


def X(board):
    s = score(board)
    if s == 10:
        printBoard(board)
        return s

    # get list of empty cells in the board (moves)
    moves = []
    for i in range(len(board)):
        # if cell is empty
        if board[i] == '-':
            moves.append(i)

    scores = []
    for m in moves:
        board[m] = 'X'
        s = O(board)
        scores.append(s)
        board[m] = '-'

    if len(scores) > 0:
        s = np.max(scores)
    else:
        s = 0
    return s

if __name__ == "__main__":
    count = 0
    board = ['-' for i in range(9)]
    board[4] = 'X'
    print('====================')
    b = O(board)
    print(b, count)
