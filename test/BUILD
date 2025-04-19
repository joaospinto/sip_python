load("@pip//:requirements.bzl", "requirement")

py_test(
    name = "simple_qp",
    srcs = ["simple_qp.py"],
    main = "simple_qp.py",
    srcs_version = "PY3",
    deps = [
        requirement("jax"),
        requirement("numpy"),
        requirement("pytest"),
        requirement("scipy"),
        "//sip_python",
    ],
    args = ["--capture=no"],
)
