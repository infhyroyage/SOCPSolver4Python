import cvxopt as co
import numpy as np
from math import sqrt


def solve_socp1():
	e = np.array([2., -1., -2.])
	c = co.matrix(e)

	g1 = np.array([[0, 1, 1], [-sqrt(2), 0, 0], [0, -1, 1]])
	g = [co.matrix(g1)]

	h1 = np.array([-2, -2 * sqrt(2), -2])
	h = [co.matrix(h1)]

	sol = co.solvers.socp(c, Gq=g, hq=h)
	print(sol['x'])


if __name__ == '__main__':
	solve_socp1()
