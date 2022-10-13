import numpy as np
from tqdm import tqdm


def l1_mult_update(X, W, H):

    m, n = np.shape(X)
    _, k = np.shape(W)
    X_approx = W @ H
    res_mat = np.abs(X - X_approx) + 1e-16
    new_W = (W / (((W @ H) / res_mat) @ H.transpose())) * ((X / res_mat) @ H.transpose())

    return new_W


def l1_mult_updates(X, W, H, max_iter, min_loss):

    m, n = np.shape(X)
    losses = []
    for t in tqdm(range(max_iter)):
        W = l1_mult_update(X, W, H)
        H = l1_mult_update(X.transpose(), H.transpose(), W.transpose()).transpose()
        iter_loss = loss(X, W, H)
        losses.append(iter_loss)
        if iter_loss <= min_loss:
            break

    return W, H, losses


def l2_mult_update(X, W, H):

    m, n = np.shape(X)
    _, k = np.shape(W)
    epsilon = 1e-16
    W_new = W * ((X @ H.T) / (W @ H @ H.T + epsilon))
    H_new = H * ((W_new.T @ X) / (W_new.T @ W_new @ H + epsilon))

    return W_new, H_new


def l2_mult_updates(X, W, H, max_iter, min_loss):

    losses = []
    for _ in tqdm(range(max_iter)):
        W, H = l2_mult_update(X, W, H)
        iter_loss = loss(X, W, H)
        losses.append(iter_loss)
        if iter_loss <= min_loss:
            break

    return W, H, losses


def loss(X, W, H):
    return np.linalg.norm(W @ H - X)
