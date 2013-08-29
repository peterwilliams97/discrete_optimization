#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Solve the n queens problem http://en.wikipedia.org/wiki/Eight_queens_puzzle
    using naive constraint propagation.
    
    Usage: 
        python n_queens.py <n>
    e.g.
        python n_queens.py 20
        
    NOTES:
        1 Uses NumPy and Numba so the best way to run this is to get the latest Anaconda distro
          from https://store.continuum.io/
        2 If you don't want to use Numba then remove the @autojit. 
        3 If you don't even want to use Numpy then make board a list of lists.
        4 This program only works for n with values up to about 900 (on Windows 64 bit) due to 
          stack size limitations in python. If you want to solve for largee values of n then convert
          add_queen() from a recursive to an iterative function.
        5 Alternative methods for solving n queens include
            Local search http://en.wikipedia.org/wiki/Min-conflicts_algorithm#Example
            MIP: http://scip.zib.de/download/files/scip_intro_01.pdf 
            A solver: e.g Comet: http://www.hakank.org/comet/queensn.co
"""
from __future__ import division
import numpy as np
import sys, random
from numba import autojit


def make_board(n):
    """Create a n x n chessboard
        Returns: An n x n integer NumPy array
    """    
    return np.zeros((n, n), dtype=np.int)
    

@autojit
def propagate(n, board, x, y, d):
    """Propagate the constraints that arise from adding (d = +1) or removing (d = -1) a queen to the 
        board at (x, y)
        
        n: Width of chessboard
        board: State of chessboard for current queens which does not include the one at (x, y). 
               board[a, b] = number of queens threating (a, b)
               board will be updated to include (d = +1) or exclude (d = -1) the queen at (x, y) 
                when this function returns
        x, y: Coordinates of queen to be added/removed
        d: +1 to add a queen, -1 to remove a queen
    """

    # Add d to the elements in the column and row that intersect (x, y) including (x, y)
    board[x, :] += d
    board[:, y] += d
    board[x, y] -= d

    # Add d to the elements in the diagonals that intersect (x, y) excluding (x, y)
    # This looks at a bit funny because of the current limitations of Numba but this is the 
    #  inner loop of this script so we put up with the odd looking code for the speed.
    x1 = x - 1
    y1 = y - 1
    x2 = x + 1
    y2 = y - 1
    x3 = x - 1
    y3 = y + 1
    x4 = x + 1
    y4 = y + 1       

    while x1 >= 0 and y1 >= 0:
        board[x1, y1] += d
        x1 -= 1
        y1 -= 1    

    while x2 < n and y2 >= 0:
        board[x2, y2] += d
        x2 += 1
        y2 -= 1
      
    while x3 >= 0 and y3 < n:
        board[x3, y3] += d
        x3 -= 1
        y3 += 1 
   
    while x4 < n and y4 < n:
        board[x4, y4] += d
        x4 += 1
        y4 += 1     


def add_queen(n, queens, board):
    """Add a queen to the column to right of the queens that have been placed on the board so far.
        Queens are added from left column to right. 
        This function gets called recursively with a feasible list of queens and attempts to add one 
         more queen immediately the right of those in the list.
        
        n: Width of chessboard
        queens: List of queens that have been placed so far. queens[i] = row of queen on column i
        board: State of chessboard. board[a, b] = number of queens threating (a, b)
        Returns: Valid list of n queens if one has been found, None otherwise
        
        Note: board gets modified then restored to its state at entry to this function.
    """
    # The queens are in feasible locations by design. If there are n of them so we must have a full 
    #  solution so we return it.
    if len(queens) == n:
        return queens

    # Checking rows in random order makes this run faster. Why?
    row_order = list(range(n))
    random.shuffle(row_order) 
    
    x = len(queens)
    for y in row_order:
        # Don't check threatened positions. The efficiency of this program is basically due to the
        #  pruning of the search space that results from not checking positions threatened by the
        #  existing queens.
        if board[x, y] != 0:
            continue
        
        # Add the contraints for the new queen at (x, y)
        propagate(n, board, x, y, 1)
        
        # Recurse
        valid_queens = add_queen(n, queens + [y], board)
        
        # Undo the constraints added for queen at x, y
        propagate(n, board, x, y, -1)
        
        # If we found a feasible full board deeper in the recursion then return it.
        if valid_queens:
            return valid_queens
            
    # If we got here then queens could be not be extended to a feasible n queens list 
    return None
    
    
def solve(n):
    """Solve the n queens problem using constraint propagation
        queens = solve(n) => queens[i] is the row of the queen in column i
                           where 0 <= row < n and 0 <= column < n
        Returns: List of rows for queens. 
    """
    return add_queen(n, [], make_board(n))

    
if __name__ == '__main__':    
    if len(sys.argv) != 2:
        print __doc__
        exit()
        
    n = int(sys.argv[1])
    print 'Solving %d queens problem' % n
    
    # Make the result reproducible. 
    random.seed(1337)
    
    queens = solve(n)
    
    print queens
    
    if n > 80:
        print 'Chessboard too big to display'
        exit()
        
    print '-' * 80
    for y in range(n):
        print ''.join(' Q' if x == y else ' +' for x in queens) 
        
