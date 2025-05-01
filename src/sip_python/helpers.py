import warnings

import numpy as np
from scipy import sparse as spa

from .sip_python_ext import getLnnz


_cvxopt_available = False
try:
    from cvxopt import amd, spmatrix

    _cvxopt_available = True
except ImportError:
    from scipy.sparse.csgraph import reverse_cuthill_mckee


def _get_K(P, A, G):
    # K = [ P + r1 I_x      A.T        G.T   ]
    #     [     A        -r2 * I_y      0    ]
    #     [     G            0       -r3 I_z ]

    if isinstance(P, np.ndarray):
        P = spa.csc_matrix(P)

    if isinstance(A, np.ndarray):
        A = spa.csr_matrix(A)

    if isinstance(G, np.ndarray):
        G = spa.csr_matrix(G)

    x_dim = P.shape[0]
    s_dim = G.shape[0]
    y_dim = A.shape[0]

    mod_P = spa.csc_matrix.copy(P)
    mod_P.data[:] = 1.0

    Z = spa.csc_matrix((y_dim, s_dim))

    K = spa.block_array(
        blocks=[
            [mod_P + spa.eye(x_dim), A.T, G.T],
            [A, -spa.eye(y_dim), Z],
            [G, Z.T, -spa.eye(s_dim)],
        ],
        format="coo",
    )

    return K


def _get_kkt_perm(P, A, G, verbose):
    K = _get_K(P=P, A=A, G=G)

    if _cvxopt_available:
        K_cvxopt = spmatrix(
            I=K.row,
            J=K.col,
            V=K.data,
        )
        return np.array(list(amd.order(K_cvxopt)))
    if verbose:
        warnings.warn(
            "cvxopt not installed; using reverse Cuthill-McKee (RCM) "
            "instead of approximate minimum degree (AMD)."
        )
    return reverse_cuthill_mckee(spa.csc_matrix(K))


def get_kkt_perm_inv(P, A, G, verbose=True):
    perm = _get_kkt_perm(P=P, A=A, G=G, verbose=verbose)

    perm_inv = np.empty_like(perm)
    perm_inv[perm] = np.arange(perm_inv.shape[0])

    return perm_inv


def get_kkt_and_L_nnzs(P, A, G, perm_inv):
    K = _get_K(P=P, A=A, G=G)

    permuted_K = spa.coo_matrix.copy(K)
    permuted_K.row = perm_inv[permuted_K.row]
    permuted_K.col = perm_inv[permuted_K.col]

    kkt_L_nnz = getLnnz(spa.triu(permuted_K))

    return K.nnz, kkt_L_nnz
