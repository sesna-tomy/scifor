'''
Given two strings s and t, return true if t is an anagram of s, and false otherwise.
An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase,
typically using all the original letters exactly once.
'''

class Solution(object):
    def isAnagram(self, s, t):
        for i in t:
            if t.count(i) != s.count(i):
                return False
            if len(t)!=len(s):
                return False
            if i not in s:
                return False
                break
        else:
            return True

sol = Solution()
sol.isAnagram("anagram","nagaram")
