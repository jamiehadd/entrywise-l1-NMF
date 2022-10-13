import json
import nmf
import numpy as np
import random
import datetime
from sklearn.decomposition import NMF


def baseline(size,
             magnitude,
             max_iter,
             min_loss,
             percent_nonzero=None):

    X, W_ground_truth, H_ground_truth = create_synthetic_data(
                            size,
                            magnitude,
                            percent_nonzero)

    m, k = np.shape(W_ground_truth)
    _, n = np.shape(H_ground_truth)
    W_initial, H_initial = initialize_normal(m, k, n)
    # W_initial, H_initial = add_noise(W_ground_truth, H_ground_truth)

    print("l2 norm between initial W, H and the ground truth")
    print(np.linalg.norm(W_initial - W_ground_truth))
    print(np.linalg.norm(H_initial - H_ground_truth))

    X_initial = W_initial @ H_initial
    W_l1, H_l1, l1_losses = nmf.l1_mult_updates(X,
                          W_initial,
                          H_initial,
                          max_iter,
                          min_loss)
    print("norm between L1 W, H and ground truth")
    print(np.linalg.norm(W_l1 - W_ground_truth))
    print(np.linalg.norm(H_l1 - H_ground_truth))

    print("L1 nmf X approx & X")
    print(np.linalg.norm(W_l1 @ H_l1 - X))

    W_l2, H_l2, l2_losses = nmf.l2_mult_updates(X,
                          W_initial,
                          H_initial,
                          max_iter,
                          min_loss)

    print("norm between L2 W, H and ground truth")
    print(np.linalg.norm(W_l2 - W_ground_truth))
    print(np.linalg.norm(H_l2 - H_ground_truth))

    print("L2 nmf X approx & X")
    print(np.linalg.norm(W_l2 @ H_l2 - X))

    ### compare with sklearn NMF
    model = NMF(n_components = int(size/2),
                init = 'custom',
                solver = 'mu',
                tol = 1e-20,
                max_iter = max_iter)
    W_std = model.fit_transform(X,
                                W = W_initial,
                                H = H_initial)
    H_std = model.components_

    print("norm bewteen std W, H and ground truth")
    print(np.linalg.norm(W_std - W_ground_truth))
    print(np.linalg.norm(H_std - H_ground_truth))

    print("Sklearn nmf X approx & X")
    print(np.linalg.norm(W_std @ H_std - X))

    std_loss = np.linalg.norm(
            W_std @ H_std - X)

    return l1_losses, l2_losses, std_loss, W_std, H_std, W_l2, H_l2

def vary_percent_nonzero(size, magnitude, max_iter):

    percent_nonzeros = list(map(lambda x : round(x*0.1, 2), list(range(10))))
    diffs = {"W_l1_diff": [], "H_l1_diff": [], "X_l1_diff": [],
             "W_l2_diff": [], "H_l2_diff": [], "X_l2_diff": []}

    for percent_nonzero in percent_nonzeros:
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
        diffs["W_l1_diff"].append(W_l1_diff)
        diffs["H_l1_diff"].append(H_l1_diff)
        diffs["X_l1_diff"].append(X_l1_diff)

        diffs["W_l2_diff"].append(W_l2_diff)
        diffs["H_l2_diff"].append(H_l2_diff)
        diffs["X_l2_diff"].append(X_l2_diff)

    save(diffs, "nonzero")


def vary_magnitude(percent_nonzero, size, max_iter):

    magnitudes = [10, 100, 1000]
    diffs = {"W_l1_diff": [], "H_l1_diff": [], "X_l1_diff": [],
             "W_l2_diff": [], "H_l2_diff": [], "X_l2_diff": []}

    for magnitude in magnitudes:
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
        diffs["W_l1_diff"].append(W_l1_diff)
        diffs["H_l1_diff"].append(H_l1_diff)
        diffs["X_l1_diff"].append(X_l1_diff)

        diffs["W_l2_diff"].append(W_l2_diff)
        diffs["H_l2_diff"].append(H_l2_diff)
        diffs["X_l2_diff"].append(X_l2_diff)

    save(diffs, "magnitude")
    return l2_nmf_losses


def vary_size(percent_nonzero, magnitude, max_iter):

    sizes = [10, 100]
    diffs = {"W_l1_diff": [], "H_l1_diff": [], "X_l1_diff": [],
             "W_l2_diff": [], "H_l2_diff": [], "X_l2_diff": []}

    for size in sizes:
        X_ground_truth, W_ground_truth, H_ground_truth = create_synthetic_data(
                                size,
                                magnitude,
                                percent_nonzero)

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
        diffs["W_l1_diff"].append(W_l1_diff)
        diffs["H_l1_diff"].append(H_l1_diff)
        diffs["X_l1_diff"].append(X_l1_diff)

        diffs["W_l2_diff"].append(W_l2_diff)
        diffs["H_l2_diff"].append(H_l2_diff)
        diffs["X_l2_diff"].append(X_l2_diff)

    save(diffs, "size")


def create_synthetic_data(size, magnitude, percent_nonzero=None):

    # X = W @ H
    # X.shape = (size, size)
    # W.shape = (size, size/2)
    # H.shape = (size/2, size)
    k = int(size/2)

    if percent_nonzero == None:
        W = abs(np.random.normal(size = (size, k)))
        H = abs(np.random.normal(size = (k, size)))
    else:
        W = np.zeros((size, k))
        H = np.zeros((k, size))
        num_nonzeros = list(range(int(percent_nonzero * size**2)))

        for _ in num_nonzeros:
            wi = random.randint(0, size - 1)
            wj = random.randint(0, k - 1)
            hi = random.randint(0, k - 1)
            hj = random.randint(0, size - 1)
            W[wi][wj] = random.uniform(0, magnitude)
            H[hi][hj] = random.uniform(0, magnitude)

    print(f"before blow-up:")
    print(W)
    print(H)
    # ground truth X
    X = W @ H
    """
    # blow up a few entries in W, H
    num_entries = 2
    scaler = 1000000
    for _ in range(num_entries):
        wi = random.randint(0, size - 1)
        wj = random.randint(0, size/2 - 1)
        hi = random.randint(0, size/2 - 1)
        hj = random.randint(0, size - 1)
        W[wi][wj] = random.uniform(0, magnitude) * scaler
        H[hi][hj] = random.uniform(0, magnitude) * scaler

    print(f"after blow-up:")
    print(W)
    print(H)
    """
    return X, W, H


def evaluate(W_ground_truth, H_ground_truth, W, H):

    W_diff = np.linalg.norm(abs(W_ground_truth - W)) / W.size
    H_diff = np.linalg.norm(abs(H_ground_truth - H)) / H.size
    X_ground_truth = W_ground_truth @ H_ground_truth
    X_diff = np.linalg.norm(abs(X_ground_truth - W @ H)) / X_ground_truth.size

    return W_diff, H_diff, X_diff


def add_noise(W, H):

    W_noise = W + abs(np.random.normal(size=W.shape))
    H_noise = H + abs(np.random.normal(size=H.shape))

    return W_noise, H_noise


def initialize_normal(m, k, n):

    W_initial = abs(np.random.normal(size=(m, k)))
    H_initial = abs(np.random.normal(size=(k, n)))

    return W_initial, H_initial


def save(results, fname):
    # time formating: "%Y%m%d-%H:%M:%S"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
    with open(f"../experiments/exp_{fname}_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
