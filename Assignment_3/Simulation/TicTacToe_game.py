'''
Tic-tac-toe is played by two players A and B on a 3 x 3 grid. The rules of Tic-Tac-Toe are:
Players take turns placing characters into empty squares ' '.
The first player A always places 'X' characters, while the second player B always places 'O' characters.
'X' and 'O' characters are always placed into empty squares, never on filled ones.
The game ends when there are three of the same (non-empty) character filling any row, column, or diagonal.
The game also ends if all squares are non-empty.
No more moves can be played if the game is over.
Given a 2D integer array moves where moves[i] = [rowi, coli] indicates that the ith move will be played on grid[rowi][coli]. return the winner of the game if it exists (A or B). In case the game ends in a draw return "Draw". If there are still movements to play return "Pending".
You can assume that moves is valid (i.e., it follows the rules of Tic-Tac-Toe), the grid is initially empty, and A will play first.
'''
lass Solution(object):
    def tictactoe(self, moves):
        metrix =[[1,2,3],
                 [4,5,6],
                 [7,8,9]]
        n = len(moves)
        for i in range(0,n,2):
            print(moves[i])
            row,col = moves[i]
            metrix[row][col] = "A"
        
        for i in range(1,n,2):
            print(moves[i])
            row,col = moves[i]
            metrix[row][col] = "B"
        if metrix[0][0] == metrix[1][1] == metrix[2][2]:
          return metrix[0][0]
        if metrix[0][2] == metrix[1][1] == metrix[2][0]:
          return metrix[0][2]
        for i in range(0,3):
          if metrix[i][0] == metrix[i][1] == metrix[i][2]:
            return metrix[i][0]
          if metrix[0][i] == metrix[1][i] == metrix[2][i]:
            return metrix[0][i]
        if n != 9:
            return "Pending"
        return "Draw"


sol = Solution()
sol.tictactoe([[0,0],[1,1],[0,1],[0,2],[1,0],[2,0]])
