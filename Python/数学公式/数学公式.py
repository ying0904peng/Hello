import math
import numpy as np
import latexify

print(latexify.__version__)


# # @latexify.function
# @latexify.expression
# def solve(a, b, c):
#     return (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


# @latexify.function
# def sinc(x):
#     if x == 0:
#         return 1
#     else:
#         return math.sin(x)/x


# @latexify.function(reduce_assignments=True, use_math_symbols=True)
# def transform(x, y, a, b, theta, s, t):
#     cos_t = math.cos(theta)
#     sin_t = math.sin(theta)
#     scale = np.array([[a, 0, 0], [0, b, 0], [0, 0, 1]])
#     rotate = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
#     move = np.array([[1, 0, s], [0, 1, t], [0, 0, 1]])
#     return move @ rotate @ scale @ np.array([[x], [y], [1]])


# @latexify.algorithmic
# def collatz(x):
#     n = 0
#     while x > 1:
#         n = n + 1
#         if x % 2 == 0:
#             x = x // 2
#         else:
#             x = 3 * x + 1
#     return n
#

@latexify.algorithmic
def fib(x):
    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return fib(x - 1) + fib(x - 2)


print(fib)
