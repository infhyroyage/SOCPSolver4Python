import cvxopt as co
import numpy as np
from math import sqrt
from random import random
from time import time
import matplotlib.pyplot as plt


def solve_socp_feasible():
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


def compare_n_socp():
    m = 20
    n_min = 5
    n_max = 35
    dn = 5

    limit_times = 100
    all_result = []

    co.solvers.options['show_progress'] = False

    for n in range(n_min, n_max + dn, dn):
        print()
        print("------ n = " + str(n) + ", m = " + str(m) + " ------")

        times = 0
        result = []

        while times < limit_times:
            threshold = np.random.rand(n)
            c = co.matrix(np.array(
                [0.0 if threshold[i] > 0.75 else 20.0 * random() - 10.0 for i in range(n)]
            ))

            threshold = [[[1.0 if i > 0 and i - 1 != j else random() for j in range(n)] for i in range(n + 1)] for _ in
                         range(m)]
            g = [co.matrix(np.array(
                [[0.0 if threshold[i][j][k] > 0.75 else 20.0 * random() - 10.0 for k in range(n)] for j in range(n + 1)]
            )) for i in range(m)]

            threshold = np.array([[1.0 if i > 0 else random() for i in range(n + 1)] for _ in range(m)])
            h = [co.matrix(np.array(
                [0.0 if threshold[i][j] > 0.75 else 20.0 * random() - 10.0 for j in range(n + 1)]
            )) for i in range(m)]

            try:
                start = time()
                sol = co.solvers.socp(c, Gq=g, hq=h)
                goal = time()

                if sol['status'] == 'optimal':
                    result.append((goal - start) * 1000)
                    times += 1
                    print(str(times) + "/" + str(limit_times) + " : " + str((goal - start) * 1000) + "(ms)")
            except ValueError:
                print(" -> Retry : Rank(A) is lesser than the number of A's rows or Rank([g[i]; A]) is lesser than n")

        all_result.append(sum(result) / len(result))
        print(" -> Finish : Average = " + str(sum(result) / len(result)) + "(ms)")

    fig = plt.figure(figsize=(14, 9))

    plt.bar(range(n_min, n_max + dn, dn), all_result, align='center')
    plt.xticks(range(n_min, n_max + dn, dn), range(n_min, n_max + dn, dn))

    plt.title("$m = " + str(m) + "$", fontsize=30)
    plt.xlabel("$n$", fontsize=30)
    plt.ylabel("CPU Time (ms)", fontsize=30)
    plt.tick_params(labelsize=25)

    fig.savefig("SOCP_n" + str(n_min) + "-" + str(n_max) + "_m" + str(m) + ".png", dpi=60)
