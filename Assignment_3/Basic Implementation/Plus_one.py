'''
You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. 
The large integer does not contain any leading 0's.
Increment the large integer by one and return the resulting array of digits.
'''

class Solution(object):
    def plusOne(self, digits):
        number = "".join([str(a) for a in digits])
        number = str(int(number)+1)
        result = [int(number[i]) for i in range(len(number))]
        return result

sol = Solution()
sol.plusOne([4,3,2,1])
