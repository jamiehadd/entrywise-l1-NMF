import nmf
import numpy as np
import random


def create_synthetic_data(size, percent_nonzero, magnitude):

    # X = W @ H
    # X.shape = (size, size)
    # W.shape = (size, size/2)
    # H.shape = (size/2, size)
    W = np.zeros((size, int(size/2)))
    H = np.zeros((int(size/2), size))
    num_nonzeros = list(range(int(percent_nonzero * size**2)))

    for _ in num_nonzeros:
        wi = random.randint(0, size - 1)
        wj = random.randint(0, size/2 - 1)
        hi = random.randint(0, size/2 - 1)
        hj = random.randint(0, size - 1)
        W[wi][wj] = random.uniform(0, magnitude)
        H[hi][hj] = random.uniform(0, magnitude)

    X = W @ H
    return X, W, H


def evaluate(W_ground_truth, H_ground_truth, W, H):

    W_diff = np.linalg.norm(abs(W_ground_truth - W)) / W.size
    H_diff = np.linalg.norm(abs(H_ground_truth - H)) / H.size
    X_ground_truth = W_ground_truth @ H_ground_truth
    X_diff = np.linalg.norm(abs(X_ground_truth - W @ H)) / X_ground_truth.size

    return W_diff, H_diff, X_diff


def add_noise(W, H):

    W_noise = W + np.random.normal(size=W.shape)
    H_noise = H + np.random.normal(size=H.shape)

    return W_noise, H_noise


def initialize_normal(m, k, n):

    W_initial = np.random.normal(size=(m, k))
    H_initial = np.random.normal(size=(k, n))

    return W_initial, H_initial


def vary_percent_zero(size, magnitude, max_iter):

    percent_nonzeros = list(map(lambda x : round(x*0.1, 2), list(range(10))))
    diffs = {"W_l1_diff": [], "H_l1_diff": [], "X_l1_diff": [],
             "W_l2_diff": [], "H_l2_diff": [], "X_l2_diff": []}
    all_diffs = {}

    for percent_nonzero in percent_nonzeros:
        all_diffs[percent_nonzero] = diffs
        X_ground_truth, W_ground_truth, H_ground_truth = create_synthetic_data(
                                size,
                                percent_nonzero,
                                magnitude)

        m, k = np.shape(w_ground_truth)
        _, n = np.shape(h_ground_truth)

        #W_initial, H_initial = initialize_normal(m, k, n)
        W_initial, H_initial = add_noise(W_ground_truth, H_ground_truth)
        X_initial = W_initial @ H_initial
        W_l1, H_l1 = nmf.l1_mult_updates(X_initial,
                              W_initial,
                              H_initial,
                              max_iter)
        W_l2, H_l2 = nmf.l2_mult_updates(X_initial,
                              W_initial,
                              H_initial,
                              max_iter)

        W_l1_diff, H_l1_diff, X_l1_diff = evaluate(W_ground_truth,
                                          H_ground_truth,
                                          W_l1,
                                          H_l1)
        W_l2_diff, H_l2_diff, X_l2_diff = evaluate(W_ground_truth,
                                          H_ground_truth,
                                          W_l2,
                                          H_l2)
        all_diffs[percent_nonzero]["W_l1_diff"].append(W_l1_diff)
        all_diffs[percent_nonzero]["H_l1_diff"].append(H_l1_diff)
        all_diffs[percent_nonzero]["X_l1_diff"].append(X_l1_diff)

        all_diffs[percent_nonzero]["W_l2_diff"].append(W_l2_diff)
        all_diffs[percent_nonzero]["H_l2_diff"].append(H_l2_diff)
        all_diffs[percent_nonzero]["X_l2_diff"].append(X_l2_diff)
        print(all_diffs[percent_nonzero])

    return all_diffs


def vary_magnitude(percent_zero, size, max_iter):

    magnitudes = [10, 100, 1000]
    diffs = {"W_l1_diff": [], "H_l1_diff": [], "X_l1_diff": [],
             "W_l2_diff": [], "H_l2_diff": [], "X_l2_diff": []}
    all_diffs = {}

    for magnitude in magnitudes:
        all_diffs[magnitude] = diffs
        X_ground_truth, W_ground_truth, H_ground_truth = create_synthetic_data(
                                size,
                                percent_nonzero,
                                magnitude)

        m, k = np.shape(W_ground_truth)
        _, n = np.shape(H_ground_truth)

        #W_initial, H_initial = initialize_normal(m, k, n)
        W_initial, H_initial = add_noise(W_ground_truth, H_ground_truth)
        X_initial = W_initial @ H_initial
        W_l1, H_l1 = nmf.l1_mult_updates(X_initial,
                              W_initial,
                              H_initial,
                              max_iter)
        W_l2, H_l2 = nmf.l2_mult_updates(X_initial,
                              W_initial,
                              H_initial,
                              max_iter)

        W_l1_diff, H_l1_diff, X_l1_diff = evaluate(W_ground_truth,
                                          H_ground_truth,
                                          W_l1,
                                          H_l1)
        W_l2_diff, H_l2_diff, X_l2_diff = evaluate(W_ground_truth,
                                          H_ground_truth,
                                          W_l2,
                                          H_l2)
        all_diffs[magnitude]["W_l1_diff"].append(W_l1_diff)
        all_diffs[magnitude]["H_l1_diff"].append(H_l1_diff)
        all_diffs[magnitude]["X_l1_diff"].append(X_l1_diff)

        all_diffs[magnitude]["W_l2_diff"].append(W_l2_diff)
        all_diffs[magnitude]["H_l2_diff"].append(H_l2_diff)
        all_diffs[magnitude]["X_l2_diff"].append(X_l2_diff)
        print(all_diffs[magnitude])

    return all_diffs


def vary_size(percent_nonzero, magnitude, max_iter):

    sizes = [10, 100]
    diffs = {"W_l1_diff": [], "H_l1_diff": [], "X_l1_diff": [],
             "W_l2_diff": [], "H_l2_diff": [], "X_l2_diff": []}
    all_diffs = {}

    for size in sizes:
        all_diffs[size] = diffs
        X_ground_truth, W_ground_truth, H_ground_truth = create_synthetic_data(
                                size,
                                percent_nonzero,
                                magnitude)

        m, k = np.shape(W_ground_truth)
        _, n = np.shape(H_ground_truth)

        #W_initial, H_initial = initialize_normal(m, k, n)
        W_initial, H_initial = add_noise(W_ground_truth, H_ground_truth)
        X_initial = W_initial @ H_initial
        W_l1, H_l1 = nmf.l1_mult_updates(X_initial,
                              W_initial,
                              H_initial,
                              max_iter)
        W_l2, H_l2 = nmf.l2_mult_updates(X_initial,
                              W_initial,
                              H_initial,
                              max_iter)

        W_l1_diff, H_l1_diff, X_l1_diff = evaluate(W_ground_truth,
                                          H_ground_truth,
                                          W_l1,
                                          H_l1)
        W_l2_diff, H_l2_diff, X_l2_diff = evaluate(W_ground_truth,
                                          H_ground_truth,
                                          W_l2,
                                          H_l2)
        all_diffs[size]["W_l1_diff"].append(W_l1_diff)
        all_diffs[size]["H_l1_diff"].append(H_l1_diff)
        all_diffs[size]["X_l1_diff"].append(X_l1_diff)

        all_diffs[size]["W_l2_diff"].append(W_l2_diff)
        all_diffs[size]["H_l2_diff"].append(H_l2_diff)
        all_diffs[size]["X_l2_diff"].append(X_l2_diff)
        print(all_diffs[size])

    return all_diffs


if __name__ == "__main__":

    size = 10
    percent_nonzero = 0.5
    magnitude = 10
    max_iter = 100000
    all_diffs = vary_size(percent_nonzero, magnitude, max_iter)

