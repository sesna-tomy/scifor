'''
Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.
Note: You must not use any built-in BigInteger library or convert the inputs to integer directly.
'''
class Solution(object):
    def multiply(self, num1, num2):
        number1 = 0
        number2 = 0
        for i in num1:
            number1 = number1 * 10 +(ord(i) - ord('0'))
        for i in num2:
            number2 = number2 * 10 + (ord(i)-ord('0')) 

        product = number1 * number2
        return str(product)


sol = Solution()
sol.multiply("10","3")
