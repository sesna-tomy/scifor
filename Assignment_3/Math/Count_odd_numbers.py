'''
Given two non-negative integers low and high. Return the count of odd numbers between low and high (inclusive).
'''

class Solution(object):
    def countOdds(self, low, high):
        count = (high - low)//2
        if high %2 !=0 or low % 2 !=0:
            count += 1

        return count

sol = Solution()
sol.countOdds(8,10)
