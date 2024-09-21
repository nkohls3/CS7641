# expressions we plan to use:
# 1 + 1 = 2
# y = m x + b
# A^2 + B^2 = C^2
# cos^2(x) + sin^2(x) = 1
# H(x) = - \sum p(x) log(p(x)) [ entropy equation ]

from latex_writer import *

height = 10
exp_height = height + 5 + 1

writer = LaTeXWriter()

# 1 + 1 = 2
# Equation 1
one_1 = Character("1", [1, height])
plus = Character("+", [2, height])
one_2 = Character("1", [3, height])
equals = Character("=", [4, height])
two = Character("2", [5, height])

eq1 = [one_1, plus, one_2, equals, two]

string_eq1 = writer.write_equation(eq1)
print("Equation 1:")
print(string_eq1)
print(" ")
print("\\bigskip")

# y = m x + b
# Equation 2
y = Character("y", [1, height])
equals = Character("=", [2, height])
m = Character("m", [3, height])
x = Character("x", [4, height])
plus = Character("+", [5, height])
b = Character("b", [6, height])

eq2 = [y, equals, m, x, plus, b]
string_eq2 = writer.write_equation(eq2)
print("Equation 2:")
print(string_eq2)
print(" ")
print("\\bigskip")

# A^2 + B^2 = C^2
# Equation 3
A = Character("A", [1, height])
two_1 = Character("2", [2, exp_height]) # change height?
plus = Character("+", [3, height])
B = Character("B", [4, height])
two_2 = Character("2", [5, exp_height]) # change height?
equals = Character("=", [6, height])
C = Character("C", [7, height])
two_3 = Character("2", [8, exp_height]) # change height?

eq3 = [A, two_1, plus, B, two_2, equals, C, two_3]
string_eq3 = writer.write_equation(eq3)
print("Equation 3:")
print(string_eq3)
print(" ")
print("\\bigskip")

# cos^2(x) + sin^2(x) = 1
# Equation 4
c = Character("c", [0, height])
o = Character("o", [1, height])
s_1 = Character("s", [2, height])
two_1 = Character("2", [3, exp_height])
open_1 = Character("(", [4, height])
x_1 = Character("x", [5, height])
close_1 = Character(")", [6, height])
plus = Character("+", [7, height])
s_2 = Character("s", [8, height])
i = Character("i", [9, height])
n = Character("n", [10, height])
two_2 = Character("2", [11, exp_height])
open_2 = Character("(", [12, height])
x_2 = Character("x", [13, height])
close_2 = Character(")", [14, height])
equals = Character("=", [15, height])
one = Character("1", [16, height])

eq4 = [c, o, s_1, two_1, open_1, x_1, close_1, plus, s_2, i, n, two_2, open_2, x_2, close_2, equals, one]
string_eq4 = writer.write_equation(eq4)
print("Equation 4:")
print(string_eq4)
print(" ")
print("\\bigskip")

# H(x) = - \sum p(x) log(p(x)) [ entropy equation ]
# Equation 5:
H = Character("H", [0, height])
open_1 = Character("(", [1, height])
x_1 = Character("x", [2, height])
close_1 = Character(")", [3, height])
equals = Character("=", [4, height])
minus = Character("-", [5, height])
sigma = Character("sum", [6, height])
p_1 = Character("p", [7, height])
open_2 = Character("(", [8, height])
x_2 = Character("x", [9, height])
close_2 = Character(")", [10, height])
l = Character("l", [11, height])
o = Character("o", [12, height])
g = Character("g", [13, height])
open_3 = Character("(", [14, height])
p_2 = Character("p", [15, height])
open_4 = Character("(", [16, height])
x_3 = Character("x", [17, height])
close_3 = Character(")", [18, height])
close_4 = Character(")", [19, height])

eq5 = [H, open_1, x_1, close_1, equals, minus, sigma, p_1, open_2, x_2, close_2, l, o, g, open_3, p_2, open_4, x_3, close_3, close_4]
string_eq5 = writer.write_equation(eq5)
print("Equation 5:")
print(string_eq5)
print(" ")
print("\\bigskip")
