# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 22:32:02 2021

@author: Lenovo
"""

# calculate P(A|B) given P(A), P(B|A), P(B|not A)
def bayes_theorem(p_a, p_b_given_a, p_not_b_given_not_a):
	# calculate P(not A)
	not_a = 1 - p_a
	# calculate P(B|not A)
	p_b_given_not_a = 1 - p_not_b_given_not_a
	# calculate P(B)
	p_b = p_b_given_a * p_a + p_b_given_not_a * not_a
	# calculate P(A|B)
	p_a_given_b = (p_b_given_a * p_a) / p_b
	return p_a_given_b
 
    
#P(A|B) = P(B|A) * P(A) / P(B)
#P(Öl|Düş) = P(Düş|Öl) * P(Öl) / P(Düş)

# P(A)
p_a =float(input("p_a değerini giriniz:"))/100 
print(p_a)
# P(B|A)
p_b_given_a = float(input("Verilen a değerindeki p_b değerini giriniz:"))/100 
print(p_b_given_a)
# P(B|not A)
p_b_given_not_a = float(input("A olmayan p_b değerini giriniz:"))/100 
print(p_b_given_not_a)
# calculate P(A|B)
result = bayes_theorem(p_a, p_b_given_a, p_b_given_not_a)
# summarize
print('P(A|B) = %.3f%%' % (result * 100))