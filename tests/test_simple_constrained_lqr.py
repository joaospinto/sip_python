import pytest
import jax
from jax import numpy as jnp

import numpy as np
from scipy import sparse as sp

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

jax.config.update("jax_enable_x64", True)


def test_simple_constrained_lqr():
    ss = Settings()
    ss.max_kkt_violation = 1e-6
    ss.enable_elastics = True
    ss.elastic_var_cost_coeff = 1e6
    ss.assert_checks_pass = True
    ss.penalty_parameter_increase_factor = 2.0
    ss.mu_update_factor = 0.9

    x_dim = 2
    u_dim = 1
    g_dim = 2
    c_dim = 1
    T = 100

    dt = 0.1

    @jax.jit
    def split(x):
        x = jnp.concatenate([x, jnp.zeros(u_dim)])
        x = x.reshape([T + 1, x_dim + u_dim])
        X = x[:, :x_dim]
        U = x[:T, x_dim:]
        return X, U

    @jax.jit
    def f(x):
        X, U = split(x)
        stagewise_costs = jax.vmap(
            lambda i: 0.5 * X[i, 0] ** 2
            + 0.5 * 0.1 * X[i, 1] ** 2
            + 0.5 * 0.1 * U[i, 0] ** 2
        )(jnp.arange(T))
        terminal_cost = 0.5 * X[T, 0] ** 2 + 0.5 * 0.1 * X[T, 1] ** 2
        return stagewise_costs.sum() + terminal_cost

    @jax.jit
    def c(x):
        x_0 = jnp.array([0.0, 10.0])
        A = jnp.array([[1.0, dt], [0.0, 1.0]])
        B = jnp.array([[dt**2 / 2], [dt]])
        X, U = split(x)

        out = jnp.concatenate(
            [
                x_0 - X[0],
                jnp.array(
                    [
                        0.0,
                    ]
                ),
                jax.vmap(
                    lambda i: jnp.concatenate(
                        [
                            (A @ X[i] + B @ U[i] - X[i + 1]),
                            jnp.array(
                                [
                                    jnp.where(i + 1 == T, X[i + 1, 1], 0.0),
                                ]
                            ),
                        ]
                    )
                )(jnp.arange(T)).flatten(),
            ]
        )
        return out

    @jax.jit
    def g(x):
        _X, U = split(x)
        return jnp.concatenate(
            [
                jax.vmap(lambda i: jnp.array([U[i, 0] - 2.0, -U[i, 0] - 2.0]))(
                    jnp.arange(T)
                ).flatten(),
                jnp.zeros(g_dim),
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

    pd = ProblemDimensions()
    pd.x_dim = T * (x_dim + u_dim) + x_dim
    pd.s_dim = (T + 1) * g_dim
    pd.y_dim = (T + 1) * (x_dim + c_dim)

    mock_x = jnp.ones(
        [
            pd.x_dim,
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
