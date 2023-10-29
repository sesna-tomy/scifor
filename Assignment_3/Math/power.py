'''
Implement pow(x, n), which calculates x raised to the power n (i.e., xn).
'''
class Solution(object):
    def myPow(self, x, n):
        if n < 0:
            n = -n
            x = 1/x 
        if n == 0:
            return 1
        elif n > 0:
            if n %2 :
                power = x * self.myPow(x,n-1)
            else:
                power = self.myPow(x*x,n/2)
                
        return power


sol = Solution()
sol.myPow(x = 2.00000, n = 10)
