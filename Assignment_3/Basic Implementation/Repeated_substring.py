'''
Given a string s, check if it can be constructed by taking a substring of it and 
appending multiple copies of the substring togethe
'''

lass Solution(object):
    def repeated(self, s):
        if s in s[1:]+s[:-1]:
            return True
        else:
            return False

sol = Solution()
sol.repeated("abab")
