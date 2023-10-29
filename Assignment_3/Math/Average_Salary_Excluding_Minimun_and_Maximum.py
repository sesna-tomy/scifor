'''
You are given an array of unique integers salary where salary[i] is the salary of the ith employee.
Return the average salary of employees excluding the minimum and maximum salary. Answers within 10-5 of the actual answer will be accepted.
'''
lass Solution(object):
    def average(self, salary):
       salary.sort()
       salary.pop()
       salary.pop(0)
       s=0.0
       for i in salary:
           s += i
       average = s / len(salary)
       return average


sol = Solution()
sol.average([4000,3000,1000,2000])
