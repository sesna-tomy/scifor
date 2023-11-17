'''
You are given an m x n integer grid accounts where accounts[i][j] is the amount of money the i​​​​​​​​​​​th​​​​ customer has in the j​​​​​​​​​​​th​​​​ bank. 
Return the wealth that the richest customer has.
A customer's wealth is the amount of money they have in all their bank accounts. 
The richest customer is the customer that has the maximum wealth.
'''

class Solution(object):
    def maximumWealth(self, accounts):
        amount =[]
        for i in accounts:
            amount.append(sum(i))
        return max(amount)
        
    

sol =Solution()
wealth = sol.maximumWealth([[1,2,3],[3,2,1]])
print(wealth)


