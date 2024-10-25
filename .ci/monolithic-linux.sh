#!/usr/bin/env bash
#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

#
# This script performs a monolithic build of the monorepo and runs the tests of
# most projects on Linux. This should be replaced by per-project scripts that
# run only the relevant tests.
#

set -ex
set -o pipefail

MONOREPO_ROOT="${MONOREPO_ROOT:="$(git rev-parse --show-toplevel)"}"
BUILD_DIR="${BUILD_DIR:=${MONOREPO_ROOT}/build}"
INSTALL_DIR="${BUILD_DIR}/install"
rm -rf "${BUILD_DIR}"

ccache --zero-stats

if [[ -n "${CLEAR_CACHE:-}" ]]; then
  echo "clearing cache"
  ccache --clear
fi

function at-exit {
  python3 "${MONOREPO_ROOT}"/.ci/generate_test_report.py ":linux: Linux x64 Test Results" \
    "linux-x64-test-results" "${BUILD_DIR}"/*-test-results.xml

  mkdir -p artifacts
  ccache --print-stats > artifacts/ccache_stats.txt
}
trap at-exit EXIT

projects="${1}"
targets="${2}"

echo "--- cmake"
pip install -q -r "${MONOREPO_ROOT}"/mlir/python/requirements.txt
pip install -q -r "${MONOREPO_ROOT}"/lldb/test/requirements.txt
pip install -q junitparser==3.2.0
cmake -S "${MONOREPO_ROOT}"/llvm -B "${BUILD_DIR}" \
      -D LLVM_ENABLE_PROJECTS="${projects}" \
      -G Ninja \
      -D CMAKE_BUILD_TYPE=Release \
      -D LLVM_ENABLE_ASSERTIONS=ON \
      -D LLVM_BUILD_EXAMPLES=ON \
      -D COMPILER_RT_BUILD_LIBFUZZER=OFF \
      -D LLVM_LIT_ARGS="-v --xunit-xml-output ${BUILD_DIR}/test-results.xml --timeout=1200 --time-tests" \
      -D LLVM_ENABLE_LLD=ON \
      -D CMAKE_CXX_FLAGS=-gmlt \
      -D LLVM_CCACHE_BUILD=ON \
      -D MLIR_ENABLE_BINDINGS_PYTHON=ON \
      -D CMAKE_INSTALL_PREFIX="${INSTALL_DIR}"

echo "--- ninja"

set +e
err_code=0
for target in $targets; do
  ninja -C "${BUILD_DIR}" -k 0 ${target}
  new_err_code=$?
  if [[ $new_err_code -ne 0 ]]; then
    err_code=${new_err_code}
  fi
  # Move so the next lit run does not overwrite it. This can fail if ninja failed
  # to get to tests, but this script will not exit because of that.
  mv "${BUILD_DIR}/test-results.xml" "${BUILD_DIR}/${target}-test-results.xml"
done

if [[ $err_code -ne 0 ]]; then
 exit $err_code
fi
set -e


# TODO: apply to runtimes also

# runtimes="${3}"
# runtime_targets="${4}"
# 
# # Compiling runtimes with just-built Clang and running their tests
# # as an additional testing for Clang.
# if [[ "${runtimes}" != "" ]]; then
#   if [[ "${runtime_targets}" == "" ]]; then
#     echo "Runtimes to build are specified, but targets are not."
#     exit 1
#   fi
# 
#   echo "--- ninja install-clang"
# 
#   ninja -C ${BUILD_DIR} install-clang install-clang-resource-headers
# 
#   RUNTIMES_BUILD_DIR="${MONOREPO_ROOT}/build-runtimes"
#   INSTALL_DIR="${BUILD_DIR}/install"
#   mkdir -p ${RUNTIMES_BUILD_DIR}
# 
#   echo "--- cmake runtimes C++03"
# 
#   cmake -S "${MONOREPO_ROOT}/runtimes" -B "${RUNTIMES_BUILD_DIR}" -GNinja \
#       -D CMAKE_C_COMPILER="${INSTALL_DIR}/bin/clang" \
#       -D CMAKE_CXX_COMPILER="${INSTALL_DIR}/bin/clang++" \
#       -D LLVM_ENABLE_RUNTIMES="${runtimes}" \
#       -D LIBCXX_CXX_ABI=libcxxabi \
#       -D CMAKE_BUILD_TYPE=RelWithDebInfo \
#       -D CMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
#       -D LIBCXX_TEST_PARAMS="std=c++03" \
#       -D LIBCXXABI_TEST_PARAMS="std=c++03"
# 
#   echo "--- ninja runtimes C++03"
# 
#   ninja -vC "${RUNTIMES_BUILD_DIR}" ${runtime_targets}
# 
#   echo "--- cmake runtimes C++26"
# 
#   rm -rf "${RUNTIMES_BUILD_DIR}"
#   cmake -S "${MONOREPO_ROOT}/runtimes" -B "${RUNTIMES_BUILD_DIR}" -GNinja \
#       -D CMAKE_C_COMPILER="${INSTALL_DIR}/bin/clang" \
#       -D CMAKE_CXX_COMPILER="${INSTALL_DIR}/bin/clang++" \
#       -D LLVM_ENABLE_RUNTIMES="${runtimes}" \
#       -D LIBCXX_CXX_ABI=libcxxabi \
#       -D CMAKE_BUILD_TYPE=RelWithDebInfo \
#       -D CMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
#       -D LIBCXX_TEST_PARAMS="std=c++26" \
#       -D LIBCXXABI_TEST_PARAMS="std=c++26"
# 
#   echo "--- ninja runtimes C++26"
# 
#   ninja -vC "${RUNTIMES_BUILD_DIR}" ${runtime_targets}
# 
#   echo "--- cmake runtimes clang modules"
# 
#   rm -rf "${RUNTIMES_BUILD_DIR}"
#   cmake -S "${MONOREPO_ROOT}/runtimes" -B "${RUNTIMES_BUILD_DIR}" -GNinja \
#       -D CMAKE_C_COMPILER="${INSTALL_DIR}/bin/clang" \
#       -D CMAKE_CXX_COMPILER="${INSTALL_DIR}/bin/clang++" \
#       -D LLVM_ENABLE_RUNTIMES="${runtimes}" \
#       -D LIBCXX_CXX_ABI=libcxxabi \
#       -D CMAKE_BUILD_TYPE=RelWithDebInfo \
#       -D CMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
#       -D LIBCXX_TEST_PARAMS="enable_modules=clang" \
#       -D LIBCXXABI_TEST_PARAMS="enable_modules=clang"
# 
#   echo "--- ninja runtimes clang modules"
#   
#   ninja -vC "${RUNTIMES_BUILD_DIR}" ${runtime_targets}
# fi
