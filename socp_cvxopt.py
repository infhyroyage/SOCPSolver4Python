import cvxopt as co
import numpy as np
from math import sqrt


def solve_socp1():
	e = np.array([-1.0, 3.0, 2.0])
	c = co.matrix(e)

	g1 = np.array([[0.0, -1.0, -sqrt(3)], [-2.0, 0.0, 0.0], [0.0, -sqrt(3), 1.0]])
	g2 = np.array([[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
	g = [co.matrix(g1), co.matrix(g2)]

	h1 = np.array([-1.0, -4.0, -sqrt(3)])
	h2 = np.array([0.0, 0.0])
	h = [co.matrix(h1), co.matrix(h2)]

	sol = co.solvers.socp(c, Gq=g, hq=h)
	print(sol['x'])


def solve_socp_infeasible():
	e = np.array([1.0, 1.0])
	c = co.matrix(e)

	g1 = np.array([[0.0, 0.0], [-1.0, 0.0], [0.0, -0.5]])
	g2 = np.array([[-0.5, 0.0], [0.0, -1.0 / sqrt(3)], [0.0, 0.0]])
	g = [co.matrix(g1), co.matrix(g2)]

	h1 = np.array([1.0, 0.0, 0.0])
	h2 = np.array([0.0, 0.0, 1.0])
	h = [co.matrix(h1), co.matrix(h2)]

	sol = co.solvers.socp(c, Gq=g, hq=h)
	print(sol['x'])


def solve_socp_unbounded():
	e = np.array([-1.0, -1.0, -1.0])
	c = co.matrix(e)

	g1 = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
	g = [co.matrix(g1)]

	h1 = np.array([0.0, 0.0, 0.0])
	h = [co.matrix(h1)]

	sol = co.solvers.socp(c, Gq=g, hq=h)
	print(sol['x'])


if __name__ == '__main__':
	print("[Infeasible SOCP]")
	solve_socp_infeasible()
	print("[Unbounded SOCP]")
	solve_socp_unbounded()
