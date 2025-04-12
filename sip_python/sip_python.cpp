#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>

#include <algorithm>

#include "sip/sip.hpp"
#include "sip_qdldl/sip_qdldl.hpp"

namespace nb = nanobind;

namespace sip_python {
auto to_sip_qdldl_spm(const sip::SparseMatrix &m)
    -> sip_qdldl::ConstSparseMatrix {
  return {m.rows, m.cols, m.ind, m.indptr, m.data, m.is_transposed};
}

auto to_sip_spm(const Eigen::SparseMatrix<double> &in, sip::SparseMatrix &out)
    -> void {
  out.rows = in.rows();
  out.cols = in.cols();
  for (int i = 0; i < out.cols; ++i) {
    out.indptr[i] = in.outerIndexPtr()[i];
  }
  for (int i = 0; i < in.nonZeros(); ++i) {
    out.ind[i] = in.innerIndexPtr()[i];
    out.data[i] = in.valuePtr()[i];
  }
  out.is_transposed = false;
}

auto to_sip_spm(const Eigen::SparseMatrix<double, Eigen::RowMajor> &in,
                sip::SparseMatrix &out) -> void {
  out.rows = in.cols();
  out.cols = in.rows();
  for (int i = 0; i <= out.cols; ++i) {
    out.indptr[i] = in.outerIndexPtr()[i];
  }
  for (int i = 0; i < in.nonZeros(); ++i) {
    out.ind[i] = in.innerIndexPtr()[i];
    out.data[i] = in.valuePtr()[i];
  }
  out.is_transposed = true;
}

struct ProblemDimensions {
  int x_dim;
  int s_dim;
  int y_dim;
  int upper_hessian_lagrangian_nnz;
  int jacobian_c_nnz;
  int jacobian_g_nnz;
  int kkt_nnz;
  int kkt_L_nnz;
  bool is_jacobian_c_transposed;
  bool is_jacobian_g_transposed;
};

struct ModelCallbackInput {
  ModelCallbackInput() = delete;

  ModelCallbackInput(const ProblemDimensions &problem_dimensions) {
    double *x_data = new double[problem_dimensions.x_dim];
    double *y_data = new double[problem_dimensions.y_dim];
    double *z_data = new double[problem_dimensions.s_dim];

    nb::capsule x_owner(x_data, [](void *p) noexcept { delete[] (double *)p; });
    nb::capsule y_owner(y_data, [](void *p) noexcept { delete[] (double *)p; });
    nb::capsule z_owner(z_data, [](void *p) noexcept { delete[] (double *)p; });

    x = nb::ndarray<nb::numpy, double, nb::ndim<1>>(
        x_data, {static_cast<unsigned long>(problem_dimensions.x_dim)},
        x_owner);
    y = nb::ndarray<nb::numpy, double, nb::ndim<1>>(
        y_data, {static_cast<unsigned long>(problem_dimensions.y_dim)},
        y_owner);
    z = nb::ndarray<nb::numpy, double, nb::ndim<1>>(
        z_data, {static_cast<unsigned long>(problem_dimensions.s_dim)},
        z_owner);
  }

  auto from(const sip::ModelCallbackInput &mci) {
    std::copy(mci.x, mci.x + x.size(), x.data());
    std::copy(mci.y, mci.y + y.size(), y.data());
    std::copy(mci.z, mci.z + z.size(), z.data());
  }

  nb::ndarray<nb::numpy, double, nb::ndim<1>> x;
  nb::ndarray<nb::numpy, double, nb::ndim<1>> y;
  nb::ndarray<nb::numpy, double, nb::ndim<1>> z;
};

struct ModelCallbackOutput {
  ModelCallbackOutput() = default;

  auto to(sip::ModelCallbackOutput &mco) const {
    mco.f = f;
    std::copy(gradient_f.data(), gradient_f.data() + gradient_f.size(),
              mco.gradient_f);
    to_sip_spm(upper_hessian_lagrangian, mco.upper_hessian_lagrangian);
    std::copy(c.data(), c.data() + c.size(), mco.c);
    to_sip_spm(jacobian_c, mco.jacobian_c);
    std::copy(g.data(), g.data() + g.size(), mco.g);
    to_sip_spm(jacobian_g, mco.jacobian_g);
  }

  double f;
  nb::ndarray<nb::numpy, double, nb::ndim<1>> gradient_f;

  Eigen::SparseMatrix<double> upper_hessian_lagrangian;

  nb::ndarray<nb::numpy, double, nb::ndim<1>> c;
  Eigen::SparseMatrix<double, Eigen::RowMajor> jacobian_c;

  nb::ndarray<nb::numpy, double, nb::ndim<1>> g;
  Eigen::SparseMatrix<double, Eigen::RowMajor> jacobian_g;
};

struct Variables {
  Variables() = delete;

  Variables(const ProblemDimensions &problem_dimensions) {
    double *x_data = new double[problem_dimensions.x_dim];
    double *s_data = new double[problem_dimensions.s_dim];
    double *e_data = new double[problem_dimensions.s_dim];
    double *y_data = new double[problem_dimensions.y_dim];
    double *z_data = new double[problem_dimensions.s_dim];

    nb::capsule x_owner(x_data, [](void *p) noexcept { delete[] (double *)p; });
    nb::capsule s_owner(s_data, [](void *p) noexcept { delete[] (double *)p; });
    nb::capsule e_owner(e_data, [](void *p) noexcept { delete[] (double *)p; });
    nb::capsule y_owner(y_data, [](void *p) noexcept { delete[] (double *)p; });
    nb::capsule z_owner(z_data, [](void *p) noexcept { delete[] (double *)p; });

    x = nb::ndarray<nb::numpy, double, nb::ndim<1>>(
        x_data, {static_cast<unsigned long>(problem_dimensions.x_dim)},
        x_owner);
    s = nb::ndarray<nb::numpy, double, nb::ndim<1>>(
        s_data, {static_cast<unsigned long>(problem_dimensions.s_dim)},
        s_owner);
    e = nb::ndarray<nb::numpy, double, nb::ndim<1>>(
        e_data, {static_cast<unsigned long>(problem_dimensions.s_dim)},
        e_owner);
    y = nb::ndarray<nb::numpy, double, nb::ndim<1>>(
        y_data, {static_cast<unsigned long>(problem_dimensions.y_dim)},
        y_owner);
    z = nb::ndarray<nb::numpy, double, nb::ndim<1>>(
        z_data, {static_cast<unsigned long>(problem_dimensions.s_dim)},
        z_owner);
  }
  nb::ndarray<nb::numpy, double, nb::ndim<1>> x;
  nb::ndarray<nb::numpy, double, nb::ndim<1>> s;
  nb::ndarray<nb::numpy, double, nb::ndim<1>> e;
  nb::ndarray<nb::numpy, double, nb::ndim<1>> y;
  nb::ndarray<nb::numpy, double, nb::ndim<1>> z;
};

struct QDLDLSettings {
  bool permute_kkt_system;
  nb::ndarray<nb::numpy, int, nb::ndim<1>> kkt_pinv;
};

using ModelCallback =
    std::function<ModelCallbackOutput(const ModelCallbackInput &)>;

class Solver {
private:
  const sip::Settings &sip_settings_;
  const sip_qdldl::Settings sip_qdldl_settings_;
  const ProblemDimensions &problem_dimensions_;
  ModelCallback model_callback_;

  ModelCallbackInput mci_;

  sip::ModelCallbackOutput sip_mco_;

  sip::Workspace workspace_;
  sip_qdldl::Workspace sip_qdldl_workspace_;

  sip_qdldl::CallbackProvider callback_provider_;

  static auto build_sip_mco_(const ProblemDimensions &problem_dimensions) {
    sip::ModelCallbackOutput mco;
    mco.reserve(problem_dimensions.x_dim, problem_dimensions.s_dim,
                problem_dimensions.y_dim,
                problem_dimensions.upper_hessian_lagrangian_nnz,
                problem_dimensions.jacobian_c_nnz,
                problem_dimensions.jacobian_g_nnz,
                problem_dimensions.is_jacobian_c_transposed,
                problem_dimensions.is_jacobian_g_transposed);
    return mco;
  }

  static auto build_workspace_(const ProblemDimensions &problem_dimensions) {
    sip::Workspace workspace;
    workspace.reserve(problem_dimensions.x_dim, problem_dimensions.s_dim,
                      problem_dimensions.y_dim, problem_dimensions.kkt_L_nnz);
    return workspace;
  }

  static auto
  build_sip_qdldl_workspace_(const ProblemDimensions &problem_dimensions) {
    sip_qdldl::Workspace sip_qdldl_workspace;
    const int kkt_dim = problem_dimensions.x_dim + problem_dimensions.y_dim +
                        problem_dimensions.s_dim;
    sip_qdldl_workspace.reserve(kkt_dim, problem_dimensions.kkt_nnz,
                                problem_dimensions.kkt_L_nnz);
    return sip_qdldl_workspace;
  }

  static auto
  build_callback_provider_(const sip_qdldl::Settings &sip_qdldl_settings,
                           const ModelCallback &model_callback,
                           ModelCallbackInput &mci,
                           sip::ModelCallbackOutput &sip_mco,
                           sip_qdldl::Workspace &sip_qdldl_workspace)
      -> sip_qdldl::CallbackProvider {
    std::fill(mci.x.data(), mci.x.data() + mci.x.size(), 0.0);
    std::fill(mci.y.data(), mci.y.data() + mci.y.size(), 0.0);
    std::fill(mci.z.data(), mci.z.data() + mci.z.size(), 0.0);
    const auto mco = model_callback(mci);
    mco.to(sip_mco);
    return sip_qdldl::CallbackProvider(
        to_sip_qdldl_spm(sip_mco.upper_hessian_lagrangian),
        to_sip_qdldl_spm(sip_mco.jacobian_c),
        to_sip_qdldl_spm(sip_mco.jacobian_g), sip_qdldl_settings,
        sip_qdldl_workspace);
  }

  static auto build_sip_qdldl_settings_(const QDLDLSettings &settings) {
    return sip_qdldl::Settings{
        .permute_kkt_system = settings.permute_kkt_system,
        .kkt_pinv = settings.kkt_pinv.data(),
    };
  }

public:
  Solver(const sip::Settings &sip_settings,
         const QDLDLSettings &sip_qdldl_settings,
         const ProblemDimensions &problem_dimensions,
         ModelCallback model_callback)
      : sip_settings_(sip_settings),
        sip_qdldl_settings_(build_sip_qdldl_settings_(sip_qdldl_settings)),
        problem_dimensions_(problem_dimensions),
        model_callback_(model_callback), mci_(problem_dimensions),
        sip_mco_(build_sip_mco_(problem_dimensions)),
        workspace_(build_workspace_(problem_dimensions)),
        sip_qdldl_workspace_(build_sip_qdldl_workspace_(problem_dimensions)),
        callback_provider_(
            build_callback_provider_(sip_qdldl_settings_, model_callback_, mci_,
                                     sip_mco_, sip_qdldl_workspace_)) {}

  ~Solver() {
    sip_qdldl_workspace_.free();
    workspace_.free();
  }

  auto solve(Variables &variables) -> sip::Output {
    std::copy(variables.x.data(), variables.x.data() + variables.x.size(),
              workspace_.vars.x);
    std::copy(variables.s.data(), variables.s.data() + variables.s.size(),
              workspace_.vars.s);
    std::copy(variables.e.data(), variables.e.data() + variables.e.size(),
              workspace_.vars.e);
    std::copy(variables.y.data(), variables.y.data() + variables.y.size(),
              workspace_.vars.y);
    std::copy(variables.z.data(), variables.z.data() + variables.z.size(),
              workspace_.vars.z);

    const auto timeout_callback = []() { return false; };

    const auto ldlt_factor =
        [this](const double *H_data, const double *C_data, const double *G_data,
               const double *w, const double r1, const double r2,
               const double r3, double *LT_data, double *D_diag) -> void {
      return callback_provider_.ldlt_factor(H_data, C_data, G_data, w, r1, r2,
                                            r3, LT_data, D_diag);
    };

    const auto ldlt_solve = [this](const double *LT_data, const double *D_diag,
                                   const double *b, double *v) -> void {
      return callback_provider_.ldlt_solve(LT_data, D_diag, b, v);
    };

    const auto add_Kx_to_y =
        [this](const double *H_data, const double *C_data, const double *G_data,
               const double *w, const double r1, const double r2,
               const double r3, const double *x_x, const double *x_y,
               const double *x_z, double *y_x, double *y_y,
               double *y_z) -> void {
      return callback_provider_.add_Kx_to_y(H_data, C_data, G_data, w, r1, r2,
                                            r3, x_x, x_y, x_z, y_x, y_y, y_z);
    };

    const auto add_upper_symmetric_Hx_to_y =
        [this](const double *H_data, const double *x, double *y) -> void {
      return callback_provider_.add_upper_symmetric_Hx_to_y(H_data, x, y);
    };

    const auto add_Cx_to_y = [this](const double *C_data, const double *x,
                                    double *y) -> void {
      return callback_provider_.add_Cx_to_y(C_data, x, y);
    };

    const auto add_CTx_to_y = [this](const double *C_data, const double *x,
                                     double *y) -> void {
      return callback_provider_.add_CTx_to_y(C_data, x, y);
    };

    const auto add_Gx_to_y = [this](const double *G_data, const double *x,
                                    double *y) -> void {
      return callback_provider_.add_Gx_to_y(G_data, x, y);
    };

    const auto add_GTx_to_y = [this](const double *G_data, const double *x,
                                     double *y) -> void {
      return callback_provider_.add_GTx_to_y(G_data, x, y);
    };

    const auto _model_callback = [&](const sip::ModelCallbackInput &mci,
                                     sip::ModelCallbackOutput **mco) -> void {
      mci_.from(mci);
      const auto _mco = model_callback_(mci_);
      _mco.to(sip_mco_);
      *mco = &sip_mco_;
    };

    sip::Input input{
        .ldlt_factor = std::cref(ldlt_factor),
        .ldlt_solve = std::cref(ldlt_solve),
        .add_Kx_to_y = std::cref(add_Kx_to_y),
        .add_upper_symmetric_Hx_to_y = std::cref(add_upper_symmetric_Hx_to_y),
        .add_Cx_to_y = std::cref(add_Cx_to_y),
        .add_CTx_to_y = std::cref(add_CTx_to_y),
        .add_Gx_to_y = std::cref(add_Gx_to_y),
        .add_GTx_to_y = std::cref(add_GTx_to_y),
        .model_callback = std::cref(_model_callback),
        .timeout_callback = std::cref(timeout_callback),
    };

    const auto output = ::sip::solve(input, sip_settings_, workspace_);

    std::copy(workspace_.vars.x, workspace_.vars.x + problem_dimensions_.x_dim,
              variables.x.data());
    std::copy(workspace_.vars.s, workspace_.vars.s + problem_dimensions_.s_dim,
              variables.s.data());
    std::copy(workspace_.vars.e, workspace_.vars.e + problem_dimensions_.s_dim,
              variables.e.data());
    std::copy(workspace_.vars.y, workspace_.vars.y + problem_dimensions_.y_dim,
              variables.y.data());
    std::copy(workspace_.vars.z, workspace_.vars.z + problem_dimensions_.s_dim,
              variables.z.data());
    return output;
  }
};
} // namespace sip_python

NB_MODULE(_sip_python, m) {
  m.doc() = "Provides Python bindings for the SIP solver.";

  nb::class_<sip_python::Solver>(m, "Solver")
      .def(nb::init<const sip::Settings &, const sip_python::QDLDLSettings &,
                    const sip_python::ProblemDimensions &,
                    sip_python::ModelCallback>(),
           nb::arg("sip_settings"), nb::arg("qdldl_settings"),
           nb::arg("problem_dimension"), nb::arg("model_callback"))
      .def("solve", &sip_python::Solver::solve);

  nb::class_<sip::Settings>(m, "Settings")
      .def(nb::init<>())
      .def_rw("max_iterations", &sip::Settings::max_iterations)
      .def_rw("max_ls_iterations", &sip::Settings::max_ls_iterations)
      .def_rw("min_iterations_for_convergence",
              &sip::Settings::min_iterations_for_convergence)
      .def_rw("num_iterative_refinement_steps",
              &sip::Settings::num_iterative_refinement_steps)
      .def_rw("max_aug_kkt_violation", &sip::Settings::max_aug_kkt_violation)
      .def_rw("max_merit_slope", &sip::Settings::max_merit_slope)
      .def_rw("tau", &sip::Settings::tau)
      .def_rw("start_ls_with_alpha_s_max",
              &sip::Settings::start_ls_with_alpha_s_max)
      .def_rw("initial_mu", &sip::Settings::initial_mu)
      .def_rw("mu_update_factor", &sip::Settings::mu_update_factor)
      .def_rw("mu_min", &sip::Settings::mu_min)
      .def_rw("initial_penalty_parameter",
              &sip::Settings::initial_penalty_parameter)
      .def_rw("min_acceptable_constraint_violation_ratio",
              &sip::Settings::min_acceptable_constraint_violation_ratio)
      .def_rw("penalty_parameter_increase_factor",
              &sip::Settings::penalty_parameter_increase_factor)
      .def_rw("penalty_parameter_decrease_factor",
              &sip::Settings::penalty_parameter_decrease_factor)
      .def_rw("max_penalty_parameter", &sip::Settings::max_penalty_parameter)
      .def_rw("armijo_factor", &sip::Settings::armijo_factor)
      .def_rw("line_search_factor", &sip::Settings::line_search_factor)
      .def_rw("line_search_min_step_size",
              &sip::Settings::line_search_min_step_size)
      .def_rw("min_merit_slope_to_skip_line_search",
              &sip::Settings::min_merit_slope_to_skip_line_search)
      .def_rw("enable_elastics", &sip::Settings::enable_elastics)
      .def_rw("elastic_var_cost_coeff", &sip::Settings::elastic_var_cost_coeff)
      .def_rw("enable_line_search_failures",
              &sip::Settings::enable_line_search_failures)
      .def_rw("print_logs", &sip::Settings::print_logs)
      .def_rw("print_line_search_logs", &sip::Settings::print_line_search_logs)
      .def_rw("print_search_direction_logs",
              &sip::Settings::print_search_direction_logs)
      .def_rw("print_derivative_check_logs",
              &sip::Settings::print_derivative_check_logs)
      .def_rw("only_check_search_direction_slope",
              &sip::Settings::only_check_search_direction_slope)
      .def_rw("assert_checks_pass", &sip::Settings::assert_checks_pass);

  nb::class_<sip_python::QDLDLSettings>(m, "QDLDLSettings")
      .def(nb::init<>())
      .def_rw("permute_kkt_system",
              &sip_python::QDLDLSettings::permute_kkt_system)
      .def_rw("kkt_pinv", &sip_python::QDLDLSettings::kkt_pinv);

  nb::enum_<sip::Status>(m, "Status")
      .value("SOLVED", sip::Status::SOLVED)
      .value("ITERATION_LIMIT", sip::Status::ITERATION_LIMIT)
      .value("LINE_SEARCH_ITERATION_LIMIT",
             sip::Status::LINE_SEARCH_ITERATION_LIMIT)
      .value("LINE_SEARCH_FAILURE", sip::Status::LINE_SEARCH_FAILURE)
      .value("TIMEOUT", sip::Status::TIMEOUT)
      .value("FAILED_CHECK", sip::Status::FAILED_CHECK)
      .export_values();

  nb::class_<sip::Output>(m, "OutputStatus")
      .def(nb::init<>())
      .def_ro("exit_status", &sip::Output::exit_status)
      .def_ro("num_iterations", &sip::Output::num_iterations)
      .def_ro("max_primal_violation", &sip::Output::max_primal_violation)
      .def_ro("max_dual_violation", &sip::Output::max_dual_violation);

  nb::class_<sip_python::ProblemDimensions>(m, "ProblemDimensions")
      .def(nb::init<>())
      .def_rw("x_dim", &sip_python::ProblemDimensions::x_dim)
      .def_rw("s_dim", &sip_python::ProblemDimensions::s_dim)
      .def_rw("y_dim", &sip_python::ProblemDimensions::y_dim)
      .def_rw("upper_hessian_lagrangian_nnz",
              &sip_python::ProblemDimensions::upper_hessian_lagrangian_nnz)
      .def_rw("jacobian_c_nnz", &sip_python::ProblemDimensions::jacobian_c_nnz)
      .def_rw("jacobian_g_nnz", &sip_python::ProblemDimensions::jacobian_g_nnz)
      .def_rw("kkt_nnz", &sip_python::ProblemDimensions::kkt_nnz)
      .def_rw("kkt_L_nnz", &sip_python::ProblemDimensions::kkt_L_nnz)
      .def_rw("is_jacobian_c_transposed",
              &sip_python::ProblemDimensions::is_jacobian_c_transposed)
      .def_rw("is_jacobian_g_transposed",
              &sip_python::ProblemDimensions::is_jacobian_g_transposed);

  nb::class_<sip_python::ModelCallbackInput>(m, "ModelCallbackInput")
      .def(nb::init<const sip_python::ProblemDimensions &>(),
           nb::arg("problem_dimensions"))
      .def_ro("x", &sip_python::ModelCallbackInput::x,
              nb::rv_policy::automatic_reference)
      .def_ro("y", &sip_python::ModelCallbackInput::y,
              nb::rv_policy::automatic_reference)
      .def_ro("z", &sip_python::ModelCallbackInput::z,
              nb::rv_policy::automatic_reference);

  nb::class_<sip_python::ModelCallbackOutput>(m, "ModelCallbackOutput")
      .def(nb::init<>())
      .def_rw("f", &sip_python::ModelCallbackOutput::f,
              nb::rv_policy::automatic_reference)
      .def_rw("gradient_f", &sip_python::ModelCallbackOutput::gradient_f,
              nb::rv_policy::automatic_reference)
      .def_rw("upper_hessian_lagrangian",
              &sip_python::ModelCallbackOutput::upper_hessian_lagrangian,
              nb::rv_policy::automatic_reference)
      .def_rw("c", &sip_python::ModelCallbackOutput::c,
              nb::rv_policy::automatic_reference)
      .def_rw("jacobian_c", &sip_python::ModelCallbackOutput::jacobian_c,
              nb::rv_policy::automatic_reference)
      .def_rw("g", &sip_python::ModelCallbackOutput::g,
              nb::rv_policy::automatic_reference)
      .def_rw("jacobian_g", &sip_python::ModelCallbackOutput::jacobian_g,
              nb::rv_policy::automatic_reference);

  nb::class_<sip_python::Variables>(m, "Variables")
      .def(nb::init<const sip_python::ProblemDimensions &>(),
           nb::arg("problem_dimensions"))
      .def_rw("x", &sip_python::Variables::x,
              nb::rv_policy::automatic_reference)
      .def_rw("s", &sip_python::Variables::s,
              nb::rv_policy::automatic_reference)
      .def_rw("e", &sip_python::Variables::e,
              nb::rv_policy::automatic_reference)
      .def_rw("y", &sip_python::Variables::y,
              nb::rv_policy::automatic_reference)
      .def_rw("z", &sip_python::Variables::z,
              nb::rv_policy::automatic_reference);
}
