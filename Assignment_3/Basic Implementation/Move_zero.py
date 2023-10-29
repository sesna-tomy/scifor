'''
Given an integer array nums, move all 0's to the end of it while maintaining 
the relative order of the non-zero elements.
Note that you must do this in-place without making a copy of the array.
'''
class Solution(object):
    def moveZeroes(self, nums):
        n = len(nums)
        p=0
        for i in nums:
            if i != 0:
                nums[p] = i
                p+=1
        while p <= n-1:
            nums[p]=0
            p+=1
        return nums

sol = Solution()
sol.moveZeroes([0,1,0,3,12])
