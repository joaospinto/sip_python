from sip_python import (
    get_kkt_and_L_nnzs,
    get_kkt_perm_inv,
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


def test_simple_qp():
    ss = Settings()
    ss.max_aug_kkt_violation = 1e-12
    ss.num_iterative_refinement_steps = 1
    ss.penalty_parameter_increase_factor = 3.0
    ss.enable_elastics = True
    ss.elastic_var_cost_coeff = 1e6
    ss.assert_checks_pass = True

    @jax.jit
    def f(x):
        return x[1] * (5.0 + x[0])

    @jax.jit
    def c(x):
        return jnp.array([])

    @jax.jit
    def g(x):
        return jnp.array(
            [
                5.0 - x[0] * x[1],
                x[0] * x[0] + x[1] * x[1] - 20.0,
            ]
        )

    @jax.jit
    def grad_f(x):
        return jax.grad(f)(x)

    @jax.jit
    def approx_upp_hess_f(x):
        def proj_psd(Q, delta=1e-6):
            S, _V = jnp.linalg.eigh(Q)
            k = -jnp.minimum(jnp.min(S), 0.0) + delta
            return Q + k * jnp.eye(Q.shape[0])

        return jnp.triu(proj_psd(jax.hessian(f)(x)))

    @jax.jit
    def jac_c(x):
        return jax.jacfwd(c)(x)

    @jax.jit
    def jac_g(x):
        return jax.jacfwd(g)(x)

    x_dim = 2

    mock_x = jnp.ones(
        [
            x_dim,
        ]
    )
    jac_c_nnz_pattern = np.array(jac_c(mock_x))
    jac_g_nnz_pattern = np.array(jac_g(mock_x))
    upper_L_hess_nnz_pattern = np.array(approx_upp_hess_f(mock_x))

    jac_c_nnz_pattern_sp = sp.csr_matrix(jac_c_nnz_pattern)
    jac_g_nnz_pattern_sp = sp.csr_matrix(jac_g_nnz_pattern)
    upper_L_hess_nnz_pattern_sp = sp.csc_matrix(upper_L_hess_nnz_pattern)

    qs = QDLDLSettings()
    qs.permute_kkt_system = True
    qs.kkt_pinv = get_kkt_perm_inv(
        P=upper_L_hess_nnz_pattern_sp,
        A=jac_c_nnz_pattern_sp,
        G=jac_g_nnz_pattern_sp,
    )

    pd = ProblemDimensions()
    pd.x_dim = x_dim
    pd.s_dim = 2
    pd.y_dim = 0
    pd.upper_hessian_lagrangian_nnz = upper_L_hess_nnz_pattern_sp.nnz
    pd.jacobian_c_nnz = jac_c_nnz_pattern_sp.nnz
    pd.jacobian_g_nnz = jac_g_nnz_pattern_sp.nnz
    pd.kkt_nnz, pd.kkt_L_nnz = get_kkt_and_L_nnzs(
        P=upper_L_hess_nnz_pattern_sp,
        A=jac_c_nnz_pattern_sp,
        G=jac_g_nnz_pattern_sp,
        perm_inv=qs.kkt_pinv,
    )
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

        upp_hess_L = np.array(approx_upp_hess_f(mci.x))
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
    assert vars.x[0] == pytest.approx(-1.15747396, abs=1e-6)
    assert vars.x[1] == pytest.approx(-4.31975162, abs=1e-6)
