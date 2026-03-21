import ctypes
import os
import warnings

import numpy as np
from scipy import sparse as spa
from scipy.sparse.csgraph import reverse_cuthill_mckee

from .sip_python_ext import getLnnz

# Try to load libamd directly for fast AMD ordering.
_libamd = None
try:
    _dylibs_dir = os.path.join(
        os.path.dirname(__import__("cvxopt").__file__), ".dylibs"
    )
    for f in os.listdir(_dylibs_dir):
        if f.startswith("libamd"):
            _libamd = ctypes.CDLL(os.path.join(_dylibs_dir, f))
            _libamd.amd_l_order.restype = ctypes.c_int
            _libamd.amd_l_order.argtypes = [
                ctypes.c_long,
                ctypes.POINTER(ctypes.c_long),
                ctypes.POINTER(ctypes.c_long),
                ctypes.POINTER(ctypes.c_long),
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            break
except Exception:
    pass


def _amd_order(K_csc):
    """Compute AMD ordering on a CSC matrix via libamd."""
    K_sym = (K_csc + K_csc.T).tocsc()
    K_sym.sort_indices()
    n = K_sym.shape[0]
    Ap = K_sym.indptr.astype(np.int64)
    Ai = K_sym.indices.astype(np.int64)
    perm = np.empty(n, dtype=np.int64)
    ret = _libamd.amd_l_order(
        n,
        Ap.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
        Ai.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
        perm.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
        None,
        None,
    )
    if ret != 0:
        raise RuntimeError(f"amd_l_order failed with return code {ret}")
    return perm.astype(np.intp)


def get_K(P, A, G):
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


def _get_kkt_perm(K, verbose):
    K_csc = spa.csc_matrix(K)
    if _libamd is not None:
        return _amd_order(K_csc)
    if verbose:
        warnings.warn(
            "cvxopt not installed; using reverse Cuthill-McKee (RCM) "
            "instead of approximate minimum degree (AMD)."
        )
    return reverse_cuthill_mckee(K_csc)


def get_kkt_perm_inv(K, verbose=True):
    perm = _get_kkt_perm(K, verbose)

    perm_inv = np.empty_like(perm)
    perm_inv[perm] = np.arange(perm_inv.shape[0])

    return perm_inv


def get_kkt_and_L_nnzs(K, perm_inv):
    permuted_K = spa.coo_matrix.copy(K)
    permuted_K.row = perm_inv[permuted_K.row]
    permuted_K.col = perm_inv[permuted_K.col]

    kkt_L_nnz = getLnnz(spa.triu(permuted_K))

    return K.nnz, kkt_L_nnz


def get_kkt_perm_inv_and_nnzs(P, A, G, verbose=True):
    K = get_K(P, A, G)
    perm_inv = get_kkt_perm_inv(K, verbose)
    K_nnz, kkt_L_nnz = get_kkt_and_L_nnzs(K, perm_inv)
    return perm_inv, K_nnz, kkt_L_nnz
