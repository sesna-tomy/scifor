'''
Given two binary strings a and b, return their sum as a binary string.
'''
class Solution(object):
    def addBinary(self,a, b):
        int_a = int(a, 2)
        int_b = int(b, 2)

        result = int_a + int_b
        binary_result = bin(result)[2:]

        return binary_result
