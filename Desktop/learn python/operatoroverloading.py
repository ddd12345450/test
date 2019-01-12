# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 15:04:36 2019

@author: TONG
"""

# =============================================================================
# OPERATOR OVERLOADING
# =============================================================================
# Python allows the same operator to act differently on different object types
# By default, python does not know how to operate on two defined objects if we don't teach them with operator overloading
# Here, using special functions, we define the data type using __str()__ method
# and then implement __add__() function to overload the + sign
class Point:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
        
    def __str__(self):
        return "({0},{1})".format(self.x,self.y)
    
    def __add__(self,other):
        x = self.x + other.x
        y = self.y + other.y
        return Point(x,y)
    
p1 = Point(2,3)
print("The coordinate of point one is ({0},{1})".format(p1.x,p1.y))
str(p1) #equivalent to p1.__str__()
format(p1)

p2 = Point(1,-1)
print("The coordinate of point two is ({0},{1})".format(p2.x,p2.y))
print(p1+p2)

#Operator Overloading Special Functions in Python
#Operator	Expression	Internally
#Addition	p1 + p2	p1.__add__(p2)
#Subtraction	p1 - p2	p1.__sub__(p2)
#Multiplication	p1 * p2	p1.__mul__(p2)
#Power	p1 ** p2	p1.__pow__(p2)
#Division	p1 / p2	p1.__truediv__(p2)
#Floor Division	p1 // p2	p1.__floordiv__(p2)
#Remainder (modulo)	p1 % p2	p1.__mod__(p2)
#Bitwise Left Shift	p1 << p2	p1.__lshift__(p2)
#Bitwise Right Shift	p1 >> p2	p1.__rshift__(p2)
#Bitwise AND	p1 & p2	p1.__and__(p2)
#Bitwise OR	p1 | p2	p1.__or__(p2)
#Bitwise XOR	p1 ^ p2	p1.__xor__(p2)
#Bitwise NOT	~p1	p1.__invert__()

# =============================================================================
# COMPARISON OPERATOR OVERLOADING
# =============================================================================
class Point:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
    
    def __str__(self):
        return "({0},{1})".format(self.x,self.y)
    
    def __lt__(self,other):
        self_mag = (self.x ** 2) + (self.y ** 2)
        other_mag = (other.x ** 2) + (other.y ** 2)
        return self_mag < other_mag
    
Poit(1,1) < Point(-2,-3)
#
#Comparision Operator Overloading in Python
#Operator	Expression	Internally
#Less than	p1 < p2	p1.__lt__(p2)
#Less than or equal to	p1 <= p2	p1.__le__(p2)
#Equal to
#
#p1 == p2	p1.__eq__(p2)
#Not equal to	p1 != p2	p1.__ne__(p2)
#Greater than	p1 > p2	p1.__gt__(p2)
#Greater than or equal to	p1 >= p2	p1.__ge__(p2)