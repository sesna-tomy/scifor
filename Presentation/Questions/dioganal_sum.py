'''
Given a square matrix mat, return the sum of the matrix diagonals.
Only include the sum of all the elements on the primary diagonal and all the elements on the secondary diagonal that are not part of the primary diagonal.
'''

class Solution(object):
    def diagonalSum(self, mat):
        add = 0
        n = -1
        for i in range(len(mat)):
            add += mat[i][i]
            add += mat[i][n]
            n -= 1
        if len(mat) % 2 == 1:
            add -= mat[len(mat)//2][len(mat)//2]
        return add
        


sol =Solution()
total = sol.diagonalSum([[1,2,3],[4,5,6],[7,8,9]])
print(total)