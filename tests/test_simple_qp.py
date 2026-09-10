import pytest
import jax
from jax import numpy as jnp

import numpy as np
from scipy import sparse as sp

from sip_python import (
    get_kkt_perm_inv_and_nnzs,
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


@pytest.mark.parametrize("defer_derivatives", [False, True])
@pytest.mark.parametrize("dual_s_max", [0.0, 0.01])
def test_simple_qp(defer_derivatives, dual_s_max):
    ss = Settings()
    ss.termination.max_dual_residual = 1e-6
    ss.termination.dual_residual_s_max = dual_s_max
    ss.termination.max_constraint_violation = 1e-6
    ss.termination.max_complementarity_gap = 1e-6
    ss.assert_checks_pass = True
    ss.logging.print_logs = False
    ss.logging.print_line_search_logs = False
    ss.logging.print_search_direction_logs = False
    ss.logging.print_derivative_check_logs = False

    @jax.jit
    def f(x):
        return (
            0.5 * (4.0 * x[0] * x[0] + 2.0 * x[0] * x[1] + 2.0 * x[1] * x[1])
            + x[0]
            + x[1]
        )

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

    pd = ProblemDimensions()
    pd.x_dim = x_dim
    pd.s_dim = jac_g_nnz_pattern_sp.shape[0]
    pd.y_dim = jac_c_nnz_pattern_sp.shape[0]

    qs = QDLDLSettings()
    qs.permute_kkt_system = True
    qs.kkt_pinv, pd.kkt_nnz, pd.kkt_L_nnz = get_kkt_perm_inv_and_nnzs(
        P=upper_L_hess_nnz_pattern_sp,
        A=jac_c_nnz_pattern_sp,
        G=jac_g_nnz_pattern_sp,
    )

    pd.upper_hessian_lagrangian_nnz = upper_L_hess_nnz_pattern_sp.nnz
    pd.jacobian_c_nnz = jac_c_nnz_pattern_sp.nnz
    pd.jacobian_g_nnz = jac_g_nnz_pattern_sp.nnz

    pd.is_jacobian_c_transposed = True
    pd.is_jacobian_g_transposed = True

    requests = []

    def mc(mci: ModelCallbackInput) -> ModelCallbackOutput:
        requests.append(mci.need_derivatives)
        mco = ModelCallbackOutput()

        mco.f = f(mci.x)
        mco.c = np.array(c(mci.x))
        mco.g = np.array(g(mci.x))

        if defer_derivatives and not mci.need_derivatives:
            return mco

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
    vars.z[:] = 1.0

    output = solver.solve(vars)

    assert requests[0] is True  # Constructor needs the derivative sparsity.
    assert True in requests and False in requests
    assert output.exit_status == Status.SOLVED
    assert vars.x[0] == pytest.approx(0.3, abs=1e-2)
    assert vars.x[1] == pytest.approx(0.7, abs=1e-2)
