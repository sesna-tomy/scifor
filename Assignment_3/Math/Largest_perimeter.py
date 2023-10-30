'''
Given an integer array nums, return the largest perimeter of a triangle with a non-zero area, formed from three of these lengths. If it is impossible to form any triangle of a non-zero area, return 0.
'''
lass Solution(object):
    def largestPerimeter(self, nums):
        nums.sort()
        lst = []
        for i in range(len(nums)-2):
            a,b,c = nums[i],nums[i+1],nums[i+2]
            if a+b>c:
                lst.append(a+b+c)
        if len(lst) >= 1:
            return max(lst)
        else:
            return 0

sol = Solution()
sol.largestPerimeter([2,1,2])
