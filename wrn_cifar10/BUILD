package(default_visibility = ["//visibility:public"])

py_library(
    name = "data",
    srcs = ["data.py"],
)

py_library(
    name = "model",
    srcs = ["model.py"],
)

py_library(
    name = "controller",
    srcs = ["controller.py"],
    deps = [
        ":data",
        ":model",
    ],
)

py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    deps = [
        ":controller",
        ":data",
        ":model",
    ],
)