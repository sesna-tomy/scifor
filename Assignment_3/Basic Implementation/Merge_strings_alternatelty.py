'''
You are given two strings word1 and word2. Merge the strings by adding letters in alternating order, 
starting with word1. If a string is longer than the other, append the additional letters onto the end of the merged string.
https://leetcode.com/problems/merge-strings-alternately/?envType=study-plan-v2&envId=programming-skills
'''
class Solution(object):
    def mergeAlternately(self, word1, word2):
       z=""
       i=0
       while i < len(word1) and i < len(word2):
           z+=word1[i]
           z+=word2[i]
           i+=1
       while i < len(word1):
           z+=word1[i]
           i+=1
       while i < len(word2):
           z+=word2[i]
           i+=1
       return z 

sol = Solution()
sol.mergeAlternately("abc","pqr") 
