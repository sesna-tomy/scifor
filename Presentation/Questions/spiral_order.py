'''
Given an m x n matrix, return all elements of the matrix in spiral order.
'''

class Solution(object):
    def spiralOrder(self, matrix):
        result = []
        while matrix:
            result += matrix.pop(0)
            if matrix and matrix[0]:
                for row in matrix:
                    result.append(row.pop())
            if matrix:
                result += matrix.pop()[::-1]
            if matrix and matrix[0]:
                for row in matrix[::-1]:
                    result.append(row.pop(0))
        return result
    

sol =Solution()
spiral = sol.spiralOrder([[1,2,3],[4,5,6],[7,8,9]])
print(spiral)