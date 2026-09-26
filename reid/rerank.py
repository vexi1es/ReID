"""k-reciprocal re-ranking (Zhong et al., CVPR 2017), компактная numpy-версия."""
import numpy as np


def k_reciprocal(q_emb, g_emb, k1=20, k2=6, lam=0.3):
    """Возвращает матрицу расстояний (n_q, n_g): меньше — ближе."""
    nq = len(q_emb)
    X = np.concatenate([q_emb, g_emb]).astype(np.float32)
    orig = np.clip(2 - 2 * X @ X.T, 0, None)            # квадрат евклида для L2-векторов
    orig /= orig.max(0, keepdims=True)
    n = len(X)
    rank = np.argsort(orig, axis=1).astype(np.int32)
    V = np.zeros((n, n), np.float32)

    for i in range(n):
        fwd = rank[i, :k1 + 1]
        back = rank[fwd, :k1 + 1]
        recip = fwd[np.where(back == i)[0]]
        expanded = recip
        for c in recip:
            cf = rank[c, :int(round(k1 / 2)) + 1]
            cb = rank[cf, :int(round(k1 / 2)) + 1]
            cr = cf[np.where(cb == c)[0]]
            if len(np.intersect1d(cr, recip)) > 2 / 3 * len(cr):
                expanded = np.append(expanded, cr)
        expanded = np.unique(expanded)
        w = np.exp(-orig[i, expanded])
        V[i, expanded] = w / w.sum()

    if k2 > 1:
        V = np.stack([V[rank[i, :k2]].mean(0) for i in range(n)])

    inv = [np.where(V[:, j] != 0)[0] for j in range(n)]
    jaccard = np.zeros((nq, n), np.float32)
    for i in range(nq):
        tmp = np.zeros(n, np.float32)
        nz = np.where(V[i] != 0)[0]
        for j in nz:
            tmp[inv[j]] += np.minimum(V[i, j], V[inv[j], j])
        jaccard[i] = 1 - tmp / (2 - tmp)

    return (jaccard * (1 - lam) + orig[:nq] * lam)[:, nq:]
