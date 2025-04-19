load("@nanobind_bazel//:build_defs.bzl", "nanobind_extension")
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")
load("@python_abi//:abi.bzl", "python_abi")
load("@python_platform//:platform.bzl", "python_platform")

nanobind_extension(
    name = "_sip_python",
    srcs = ["sip_python.cpp"],
    deps = [
        "@sip//sip",
        "@sip_qdldl//sip_qdldl",
        "@qdldl",
        "@eigen",
    ],
    linkstatic = True,
    visibility = ["//visibility:public",],
)

py_library(
    name = "sip_python",
    srcs = [
        "__init__.py",
    ],
    data = [":_sip_python.so"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)

py_package(
    name = "sip_python_pkg",
    visibility = ["//visibility:private"],
    deps = [
        ":sip_python",
    ],
)

py_wheel(
    name = "wheel",
    abi = python_abi(),
    author = "João Sousa-Pinto",
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    description_file = "//:README.md",
    distribution = "sip_python",
    extra_distinfo_files = {
        "//:LICENSE": "LICENSE",
    },
    homepage = "https://github.com/joaospinto/sip_python",
    license = "MIT",
    python_requires = ">=3.10",
    platform = select({
        "@bazel_tools//src/conditions:windows_x64": "win_amd64",
        "@bazel_tools//src/conditions:windows_arm64": "win_arm64",
        "@bazel_tools//src/conditions:darwin_x86_64": "macosx_14_0_x86_64",
        "@bazel_tools//src/conditions:darwin_arm64": "macosx_14_0_arm64",
        "@bazel_tools//src/conditions:linux_x86_64": python_platform() + "_x86_64",
        "@bazel_tools//src/conditions:linux_aarch64": python_platform() + "_aarch64",
    }),
    python_tag = python_abi(),
    requires = ["numpy", "scipy"],
    twine = "@publish_deps//twine",
    version = "0.0.1",
    deps = [
        ":sip_python_pkg",
    ],
)
