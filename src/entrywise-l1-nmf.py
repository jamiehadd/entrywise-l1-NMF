from tqdm import tqdm
import numpy as np

def l1_mult_update(X, W, H):

    m,n = np.shape(X)
    _,k = np.shape(W)
    X_approx = W@H
    res_mat = np.abs(X - X_approx) + 1e-10
    new_W = (W/(((W@H)/res_mat)@H.transpose()))*((X/res_mat)@H.transpose())

    return new_W


def l1_mult_updates(X,k,max_iter,W,H):
    m,n = np.shape(X)

    for t in range(max_iter):
        W = l1_mult_update(X,W,H)
        H = l1_mult_update(X.transpose(),H.transpose(),W.transpose()).transpose()

    return W,H
