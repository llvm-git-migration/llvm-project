# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")
load(":vulkan_sdk.bzl", "vulkan_sdk_setup")
load(":configure.bzl", "llvm_configure", "DEFAULT_TARGETS")

def _llvm_configure_extension_impl(ctx):
    targets = []

    # Aggregate targets across imports.
    for module in ctx.modules:
        for config in module.tags.configure:
            for target in config.targets:
                if target not in targets:
                    targets.append(target)

    new_local_repository(
        name = "llvm-raw",
        path = "../../",
        build_file_content = "#Empty.",
    )

    # Fall back to the default targets if all configurations of this extension
    # omit the `target` attribute.
    if targets == []:
        targets = DEFAULT_TARGETS

    llvm_configure(name = "llvm-project", targets = targets)

    http_archive(
        name = "vulkan_headers",
        build_file = "@llvm-raw//utils/bazel/third_party_build:vulkan_headers.BUILD",
        sha256 = "19f491784ef0bc73caff877d11c96a48b946b5a1c805079d9006e3fbaa5c1895",
        strip_prefix = "Vulkan-Headers-9bd3f561bcee3f01d22912de10bb07ce4e23d378",
        urls = [
            "https://github.com/KhronosGroup/Vulkan-Headers/archive/9bd3f561bcee3f01d22912de10bb07ce4e23d378.tar.gz",
        ],
    )

    vulkan_sdk_setup(name = "vulkan_sdk")

llvm_project_overlay = module_extension(
    doc = """Configure the llvm-project.

    Tags:
        targets: List of targets which Clang should support.
    """,
    implementation = _llvm_configure_extension_impl,
    tag_classes = {
        "configure": tag_class(
            attrs = {
                "targets": attr.string_list(),
            },
        ),
    },
)
