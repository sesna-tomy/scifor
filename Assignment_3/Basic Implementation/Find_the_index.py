'''
Given two strings needle and haystack, return the index of the first occurrence of needle in haystack, 
or -1 if needle is not part of haystack.
'''
class Solution(object):
    def strStr(self, haystack, needle):
        if needle in haystack:
            return haystack.index(needle) 
        else:
            return -1

sol = Solution()
sol.strStr("sadbutsad","sad")
