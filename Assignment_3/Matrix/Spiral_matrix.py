'''
Given an m x n matrix, return all elements of the matrix in spiral order.
'''
class Solution(object):
    def spiralOrder(self, mat):
        lst = []
        n = len(mat)+len(mat[0])
        try:
          for i in range(n): 
            for i in mat[0]:
                lst.append(i)
            mat.pop(0)
            for i in mat:
                lst.append(i[-1])
                i.pop(-1)
            lst1 = [i for i in range(len(mat[0]))]
            for k in lst[:len(lst)]:
              i = -1 
            for j in lst1[::-1]:
              lst.append(mat[i][j])
            mat.pop(i)
            for i in mat[::-1]:
                lst.append(i[0])
                i.pop(0)
        except IndexError:
          pass            

        return lst


sol = Solution()
sol.spiralOrder([[1,2,3],[4,5,6],[7,8,9]])
