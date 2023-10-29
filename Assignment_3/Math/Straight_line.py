'''
You are given an array coordinates, coordinates[i] = [x, y], where [x, y] represents the coordinate of a point. Check if these points make a straight line in the XY plane.
'''
class Solution(object):
    def check(self, coo):
       x1,y1 = coo[0]
       x2,y2 = coo[1]
       for i in range(2,len(coo)):
           x,y =coo[i]
           if (y-y1) * (x-x2) != (y-y2)*(x-x1):
               return False
        
       return True

sol = Solution()
sol.check([[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]])
