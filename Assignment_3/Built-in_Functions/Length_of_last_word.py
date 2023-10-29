'''
Given a string s consisting of words and spaces, return the length of the last word in the string.
A word is a maximal substring consisting of non-space characters only.
'''
class Solution(object):
    def length(self, s):
        dup = s.split()
        count = len(dup[-1])
        return count


sol =Solution()
sol.length("Hello World")
