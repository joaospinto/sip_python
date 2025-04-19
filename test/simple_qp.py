from sip_python import (
    ModelCallbackInput,
    ModelCallbackOutput,
    ProblemDimensions,
    QDLDLSettings,
    Settings,
    Solver,
    Status,
    Variables,
)

import pytest

import jax
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)

import numpy as np

from scipy import sparse as sp

import sys

def proj_psd(Q, delta=1e-6):
    S, V = np.linalg.eigh(Q)
    k = np.min(S) + delta
    return Q + k * np.eye(Q.shape[0])

def test_simple_qp():
    ss = Settings()
    ss.max_aug_kkt_violation = 1e-12
    ss.enable_elastics = True
    ss.elastic_var_cost_coeff = 1e6
    ss.assert_checks_pass= True

    qs = QDLDLSettings()
    qs.permute_kkt_system = True
    qs.kkt_pinv = np.arange(7)[::-1]

    @jax.jit
    def f(x):
        return 0.5 * (4.0 * x[0] * x[0] + 2.0 * x[0] * x[1] + 2.0 * x[1] * x[1]) + x[0] + x[1]
    @jax.jit
    def c(x):
        return jnp.array([x[0] + x[1] - 1.0])
    @jax.jit
    def g(x):
        return jnp.array([x[0] - 0.7, -x[0] - 0.0, x[1] - 0.7, -x[1] - 0.0])
    @jax.jit
    def grad_f(x):
        return jax.grad(f)(x)
    @jax.jit
    def hess_f(x):
        return jax.hessian(f)(x)
    @jax.jit
    def jac_c(x):
        return jax.jacfwd(c)(x)
    @jax.jit
    def jac_g(x):
        return jax.jacfwd(g)(x)

    x_dim = 2

    jac_c_nnz_pattern = np.array(jac_c(jnp.ones([x_dim,])))
    jac_g_nnz_pattern = np.array(jac_g(jnp.ones([x_dim,])))
    upp_L = proj_psd(np.array(hess_f(jnp.ones([x_dim,]))))
    upper_L_hess_nnz_pattern = np.triu(upp_L)

    jac_c_nnz_pattern_sp = sp.csr_matrix(jac_c_nnz_pattern)
    jac_g_nnz_pattern_sp = sp.csr_matrix(jac_g_nnz_pattern)
    upper_L_hess_nnz_pattern_sp = sp.csc_matrix(upper_L_hess_nnz_pattern)

    pd = ProblemDimensions()
    pd.x_dim = x_dim
    pd.s_dim = 4
    pd.y_dim = 1
    pd.upper_hessian_lagrangian_nnz = upper_L_hess_nnz_pattern_sp.nnz
    pd.jacobian_c_nnz = jac_c_nnz_pattern_sp.nnz
    pd.jacobian_g_nnz = jac_g_nnz_pattern_sp.nnz
    pd.kkt_nnz = 14
    pd.kkt_L_nnz = 15
    pd.is_jacobian_c_transposed = True
    pd.is_jacobian_g_transposed = True

    def mc(mci: ModelCallbackInput) -> ModelCallbackOutput:
        mco = ModelCallbackOutput()

        mco.f = f(mci.x)
        mco.c = np.array(c(mci.x))
        mco.g = np.array(g(mci.x))

        mco.gradient_f = np.array(grad_f(mci.x))

        C = np.array(jac_c(mci.x))
        jac_c_nnz_pattern_sp.data = C[jac_c_nnz_pattern != 0.0]
        mco.jacobian_c = jac_c_nnz_pattern_sp

        G = np.array(jac_g(mci.x))
        jac_g_nnz_pattern_sp.data = G[jac_g_nnz_pattern != 0.0]
        mco.jacobian_g = jac_g_nnz_pattern_sp

        hess_L = np.array(hess_f(mci.x))
        hess_L = proj_psd(hess_L)
        upp_hess_L = np.triu(hess_L)
        upper_L_hess_nnz_pattern_sp.data = upp_hess_L[upper_L_hess_nnz_pattern != 0.0]
        mco.upper_hessian_lagrangian = upper_L_hess_nnz_pattern_sp

        return mco

    solver = Solver(ss, qs, pd, mc)

    vars = Variables(pd)
    vars.x[:] = 0.0
    vars.s[:] = 1.0
    vars.y[:] = 0.0
    vars.e[:] = 0.0
    vars.z[:] = 1.0

    output = solver.solve(vars)

    assert output.exit_status == Status.SOLVED
    assert vars.x[0] == pytest.approx(0.3, abs=1e-5)
    assert vars.x[1] == pytest.approx(0.7, abs=1e-5)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
