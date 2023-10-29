'''
Given a square matrix mat, return the sum of the matrix diagonals.
Only include the sum of all the elements on the primary diagonal and all the elements on the secondary diagonal that are not part of the primary diagonal.
'''

class Solution(object):
    def diagonalSum(self, mat):
        add = 0
        p = len(mat)
        n=-1
        for i in range(p):
            add += mat[i][i]
            add += mat[i][n]
            n-=1
        if p % 2==1:
            add-=mat[n/2][n/2]
        return add


sol =Solution()
sol.diagonalSum([[1,2,3],
              [4,5,6],
              [7,8,9]])
