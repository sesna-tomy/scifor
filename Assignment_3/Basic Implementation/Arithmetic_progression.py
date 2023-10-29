'''
A sequence of numbers is called an arithmetic progression if the difference between any two consecutive elements is the same.
Given an array of numbers arr, return true if the array can be rearranged to form an arithmetic progression. 
Otherwise, return false.
'''
class Solution(object):
    def Make(self, arr):
        arr = sorted(arr)
        diff = arr[0]-arr[1]
        for i in range(1,len(arr)):
            are = arr[i-1]-arr[i]
            if are != diff:
                return False
                break
        return True


sol = Solution()
sol.Make([3,5,1])
