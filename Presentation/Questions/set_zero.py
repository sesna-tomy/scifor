'''
Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.
You must do it
'''

class Solution(object):
    def setZeroes(self, matrix):
        rows, cols = len(matrix), len(matrix[0])
        zero_rows, zero_cols = set(), set()

        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == 0:
                    zero_rows.add(i)
                    zero_cols.add(j)

        for row in zero_rows:
            matrix[row] = [0] * cols

        for col in zero_cols:
            for i in range(rows):
                matrix[i][col] = 0

        return matrix


sol=Solution()
matrix = sol.setZeroes([[1,1,1],[1,0,1],[1,1,1]])
print(matrix)