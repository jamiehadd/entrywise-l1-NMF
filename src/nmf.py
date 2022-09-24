import numpy as np
from tqdm import tqdm


def l1_mult_update(X, W, H):

    m, n = np.shape(X)
    _, k = np.shape(W)
    X_approx = W @ H
    res_mat = np.abs(X - X_approx) + 1e-10
    new_W = (W / (((W @ H) / res_mat) @ H.transpose())) * ((X / res_mat) @ H.transpose())

    return new_W


def l1_mult_updates(X, W, H, max_iter):

    m, n = np.shape(X)
    for t in tqdm(range(max_iter)):
        W = l1_mult_update(X, W, H)
        H = l1_mult_update(X.transpose(), H.transpose(), W.transpose()).transpose()

    return W, H


def l2_mult_update(X, W, H):

    # A -> W
    # S -> H
    m, n = np.shape(X)
    _, k = np.shape(W)
    epsilon = 1e-6
    W_new = W * ((X @ H.T) / (W @ H @ H.T + epsilon))
    H_new = H * ((W.T @ X) / (W.T @ W @ H + epsilon))

    return W_new, H_new


def l2_mult_updates(X, W, H, max_iter):

    for _ in tqdm(range(max_iter)):
        W, H = l2_mult_update(X, W, H)

    return W, H
