import numpy as np

def b_1(x, v, u, a_1, a_2, a, u_max, beta) -> float:
    return x

def b_2(x, v, u, a_1, a_2, a, u_max, beta) -> float:
    return v + a_1 * x

def h_1(x, v, u, a_1, a_2, a, u_max, beta) -> float:
    return u + (a_1 + a_2) * v + a_2*a_1*x

def h_2(x, v, u, a_1, a_2, a, u_max, beta) -> float:
    return u_max - u

def softmin(x, v, u, a_1, a_2, a, u_max, beta):

    # Compute the softmin
    return np.log(np.exp(-beta * h_1(x, v, u, a_1, a_2, a, u_max, beta)) + np.exp(-beta * h_2(x, v, u, a_1, a_2, a, u_max, beta))) / beta

def h_1_equal_h2_get_u(x, v, a_1, a_2, a, u_max, beta):

    u = (-(a_1 + a_2)*v - a_2*a_1*x + u_max)/2

    return u

def RHS(x, v, u, a_1, a_2, a, u_max, beta) -> float:
    return - ((a_1 + a_2) * u - a_2 * a_1 * v) / 2 - softmin(x, v, u, a_1, a_2, a, u_max, beta)


if __name__ == "__main__":

    # We are looking for a case were h_1 > 0, h_2 > 0 b_1 > 0, b_2 > 0, softmin > 0, and RHS > 0

    # We will sample randomly the parameters

    i = 0
    while True:

        x = np.random.uniform(0, 10)
        v = np.random.uniform(-100, 0)
        u_max = np.random.uniform(0, 10)
        a_1 = np.random.uniform(0.0001, 10)
        a_2 = np.random.uniform(0.0001, 10)
        a = np.random.uniform(0.0001, 10)
        beta = np.random.uniform(0.0001, 3)

        u = h_1_equal_h2_get_u(x, v, a_1, a_2, a, u_max, beta)

        # Check the conditions
        h1 = h_1(x, v, u, a_1, a_2, a, u_max, beta)
        h2 = h_2(x, v, u, a_1, a_2, a, u_max, beta)
        b1 = b_1(x, v, u, a_1, a_2, a, u_max, beta)
        b2 = b_2(x, v, u, a_1, a_2, a, u_max, beta)
        soft_min = softmin(x, v, u, a_1, a_2, a, u_max, beta)
        rhs = RHS(x, v, u, a_1, a_2, a, u_max, beta)

        if h1 >= 0 and h2 >= 0 and b1 >= 0 and b2 >= 0 and soft_min >= 0 and rhs > 0:
            print("Found a counter example")
            print(f"x: {x}, v: {v}, u: {u}, a_1: {a_1}, a_2: {a_2}, a: {a}, u_max: {u_max}, beta: {beta}")
            print(f"h1: {h1}, h2: {h2}, b1: {b1}, b2: {b2}, soft_min: {soft_min}, rhs: {rhs}")
            break

        print(i)

        i += 1