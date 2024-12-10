import concurrent.futures
import functools
import gc
import importlib.util
import sys
import threading
import tempfile

from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Optional

import pytest

import mlir.dialects.arith as arith
from mlir.dialects import transform
from mlir.ir import Context, Location, Module, IntegerType, F64Type, InsertionPoint


def import_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def copy_and_update(src_filepath: Path, dst_filepath: Path):
    # We should remove all calls like `run(testMethod)`
    with open(src_filepath, "r") as reader, open(dst_filepath, "w") as writer:
        while True:
            src_line = reader.readline()
            if len(src_line) == 0:
                break
            skip_lines = [
                "run(",
                "@run",
                "@constructAndPrintInModule",
                "run_apply_patterns(",
                "@run_apply_patterns",
                "@test_in_context",
                "@construct_and_print_in_module",
            ]
            if any(src_line.startswith(line) for line in skip_lines):
                continue
            writer.write(src_line)


def run(f):
    f()


def run_with_context_and_location(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        f()
    return f


def run_with_insertion_point(f):
    print("\nTEST:", f.__name__)
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f(ctx)
        print(module)


def run_with_insertion_point_v2(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)
    return f


def run_with_insertion_point_v3(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            print("\nTEST:", f.__name__)
            f(module)
        print(module)
    return f


def run_with_insertion_point_v4(f):
    print("\nTEST:", f.__name__)
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = Module.create()
        with InsertionPoint(module.body):
            f()
    return f


def run_apply_patterns(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            sequence = transform.SequenceOp(
                transform.FailurePropagationMode.Propagate,
                [],
                transform.AnyOpType.get(),
            )
            with InsertionPoint(sequence.body):
                apply = transform.ApplyPatternsOp(sequence.bodyTarget)
                with InsertionPoint(apply.patterns):
                    f()
                transform.YieldOp()
        print("\nTEST:", f.__name__)
        print(module)
    return f


def run_transform_tensor_ext(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            sequence = transform.SequenceOp(
                transform.FailurePropagationMode.Propagate,
                [],
                transform.AnyOpType.get(),
            )
            with InsertionPoint(sequence.body):
                f(sequence.bodyTarget)
                transform.YieldOp()
        print(module)
    return f


def run_transform_structured_ext(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            print("\nTEST:", f.__name__)
            f()
        module.operation.verify()
        print(module)
    return f


def run_construct_and_print_in_module(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            module = f(module)
        if module is not None:
            print(module)
    return f


# Python 3.13.1 experimental free-threading build (tags/v3.13.1:06714517797, Dec 10 2024, 00:18:06) [Clang 15.0.7 ]
# numpy      2.3.0.dev0
# nanobind   2.5.0.dev1 /tmp/jax/nanobind
# pybind11   master
test_modules = [
    # Failed tests,
    # - multithreaded_tests.py::TestAllMultiThreaded::test_execution_engine__testBF16Memref_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_execution_engine__testBasicCallback_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_execution_engine__testComplexMemrefAdd_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_execution_engine__testComplexUnrankedMemrefAdd_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_execution_engine__testDynamicMemrefAdd2D_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_execution_engine__testF16MemrefAdd_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_execution_engine__testF8E5M2Memref_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_execution_engine__testInvalidModule_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_execution_engine__testInvokeFloatAdd_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_execution_engine__testMemrefAdd_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_execution_engine__testNanoTime_multi_threaded
    ("execution_engine", run),  # Fail,

    # Failed tests,
    # - multithreaded_tests.py::TestAllMultiThreaded::test_pass_manager__testPrintIrAfterAll_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_pass_manager__testPrintIrBeforeAndAfterAll_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_pass_manager__testPrintIrLargeLimitElements_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_pass_manager__testPrintIrTree_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_pass_manager__testRunPipeline_multi_threaded
    ("pass_manager", run),  # Fail

    # Dialects tests, 8 failed, 206 passed
    # - Failing tests:
    # - TSAN unrelatd: multithreaded_tests.py::TestAllMultiThreaded::test_dialects_arith_dialect__testArithValue_multi_threaded
    #   RuntimeError: Value caster is already registered: <class 'dialects/arith_dialect.testArithValue.<locals>.ArithValue'>
    # - multithreaded_tests.py::TestAllMultiThreaded::test_dialects_memref__testSubViewOpInferReturnTypeExtensiveSlicing_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_dialects_transform_structured_ext__testMatchInterfaceEnumReplaceAttributeBuilder_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_dialects_transform_interpreter__include_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_dialects_transform_interpreter__print_other_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_dialects_transform_interpreter__print_self_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_dialects_transform_interpreter__transform_options_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_dialects_gpu_module-to-binary-rocdl__testGPUToASMBin_multi_threaded
    ("dialects/affine", run_with_insertion_point_v2),  # Pass
    ("dialects/func", run_with_insertion_point_v2),  # Pass
    ("dialects/arith_dialect", run),  # Fail
    ("dialects/arith_llvm", run),  # Pass
    ("dialects/async_dialect", run),  # Pass
    ("dialects/builtin", run),  # Pass
    ("dialects/cf", run_with_insertion_point_v4),  # Pass
    ("dialects/complex_dialect", run),  # Pass
    ("dialects/func", run_with_insertion_point_v2),  # Pass
    ("dialects/index_dialect", run_with_insertion_point),  # Pass
    ("dialects/llvm", run_with_insertion_point_v2),  # Pass
    ("dialects/math_dialect", run),  # Pass
    ("dialects/memref", run),  # Fail
    ("dialects/ml_program", run_with_insertion_point_v2),  # Pass
    ("dialects/nvgpu", run_with_insertion_point_v2),  # Pass
    ("dialects/nvvm", run_with_insertion_point_v2),  # Pass
    ("dialects/ods_helpers", run),  # Pass
    ("dialects/openmp_ops", run_with_insertion_point_v2),  # Pass
    ("dialects/pdl_ops", run_with_insertion_point_v2),  # Pass
    # ("dialects/python_test", run),  # Need to pass pybind11 or nanobind argv
    ("dialects/quant", run),  # Pass
    ("dialects/rocdl", run_with_insertion_point_v2),  # Pass
    ("dialects/scf", run_with_insertion_point_v2),  # Pass
    ("dialects/shape", run),  # Pass
    ("dialects/spirv_dialect", run),  # Pass
    ("dialects/tensor", run),  # Pass
    # ("dialects/tosa", ),  # Nothing to test
    ("dialects/transform_bufferization_ext", run_with_insertion_point_v2),  # Pass
    # ("dialects/transform_extras", ),  # Needs a more complicated execution schema
    ("dialects/transform_gpu_ext", run_transform_tensor_ext),  # Pass
    ("dialects/transform_interpreter", run_with_context_and_location, ["print_", "transform_options", "failed", "include"]),  # Fail
    ("dialects/transform_loop_ext", run_with_insertion_point_v2, ["loopOutline"]),  # Pass
    ("dialects/transform_memref_ext", run_with_insertion_point_v2),  # Pass
    ("dialects/transform_nvgpu_ext", run_with_insertion_point_v2),  # Pass
    ("dialects/transform_sparse_tensor_ext", run_transform_tensor_ext),  # Pass
    ("dialects/transform_structured_ext", run_transform_structured_ext),  # Fail
    ("dialects/transform_tensor_ext", run_transform_tensor_ext),  # Pass
    ("dialects/transform_vector_ext", run_apply_patterns, ["configurable_patterns"]),  # Pass
    ("dialects/transform", run_with_insertion_point_v3),  # Pass
    ("dialects/vector", run_with_context_and_location),  # Pass

    ("dialects/gpu/dialect", run_with_context_and_location),  # Pass
    ("dialects/gpu/module-to-binary-nvvm", run_with_context_and_location),  # Pass
    ("dialects/gpu/module-to-binary-rocdl", run_with_context_and_location),  # Fail

    ("dialects/linalg/ops", run),  # Pass
    # TO ADD:
    # ("dialects/linalg/opsdsl/*", run),  #

    ("dialects/sparse_tensor/dialect", run),  # Pass
    ("dialects/sparse_tensor/passes", run),  # Pass

    # Integration tests, 2 failed, 11 passed
    # - Failing tests:
    # - multithreaded_tests.py::TestAllMultiThreaded::test_integration_dialects_linalg_opsrun__test_elemwise_builtin_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_integration_dialects_linalg_opsrun__test_elemwise_generic_multi_threaded
    ("integration/dialects/pdl", run_construct_and_print_in_module),  # Pass
    ("integration/dialects/transform", run_construct_and_print_in_module),  # Pass
    ("integration/dialects/linalg/opsrun", run),  # Fail

    # IR tests
    # - Failing tests:
    # - multithreaded_tests.py::TestAllMultiThreaded::test_ir_debug__testDebugDlag_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_ir_diagnostic_handler__testDiagnosticCallbackException_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_ir_dialects__testAppendPrefixSearchPath_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_ir_module__testParseSuccess_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_ir_operation__testKnownOpView_multi_threaded
    # - multithreaded_tests.py::TestAllMultiThreaded::test_ir_value__testValueCasters_multi_threaded
    # - Crashed: multithreaded_tests.py::TestAllMultiThreaded::test_ir_value__testValuePrintAsOperand_multi_threaded
    ("ir/affine_expr", run),  # Pass
    ("ir/affine_map", run),  # Pass
    ("ir/array_attributes", run),  # Pass
    ("ir/attributes", run),  # Pass
    ("ir/blocks", run),  # Pass
    ("ir/builtin_types", run),  # Pass
    ("ir/context_managers", run),  # Pass
    ("ir/debug", run),  # Fail
    ("ir/diagnostic_handler", run),  # Fail
    ("ir/dialects", run),  # Fail
    ("ir/exception", run),  # Pass
    ("ir/insertion_point", run),  # Pass
    ("ir/integer_set", run),  # Pass
    ("ir/location", run),  # Pass
    ("ir/module", run),  # Pass but may fail randomly on mlirOperationDump in testParseSuccess
    ("ir/operation", run),  # Fail
    ("ir/symbol_table", run),  # Pass
    ("ir/value", run),  # Fail/Crash
]


def add_existing_tests(test_prefix: str = "_original_test"):
    def decorator(test_cls):
        this_folder = Path(__file__).parent.absolute()
        test_cls.output_folder = tempfile.TemporaryDirectory()
        output_folder = Path(test_cls.output_folder.name)

        for test_mod_info in test_modules:
            assert isinstance(test_mod_info, tuple) and len(test_mod_info) in (2, 3)
            if len(test_mod_info) == 2:
                test_module_name, exec_fn = test_mod_info
                test_pattern = None
            else:
                test_module_name, exec_fn, test_pattern = test_mod_info

            src_filepath = this_folder / f"{test_module_name}.py"
            dst_filepath = (output_folder / f"{test_module_name}.py").absolute()
            if not dst_filepath.parent.exists():
                dst_filepath.parent.mkdir(parents=True)
            copy_and_update(src_filepath, dst_filepath)
            test_mod = import_from_path(test_module_name, dst_filepath)
            for attr_name in dir(test_mod):
                is_test_fn = test_pattern is None and attr_name.startswith("test")
                is_test_fn |= test_pattern is not None and any([p in attr_name for p in test_pattern])
                if is_test_fn:
                    obj = getattr(test_mod, attr_name)
                    if callable(obj):
                        test_name = f"{test_prefix}_{test_module_name.replace('/', '_')}__{attr_name}"
                        def wrapped_test_fn(self, *args, __test_fn__=obj, __exec_fn__=exec_fn, **kwargs):
                            __exec_fn__(__test_fn__)

                        setattr(test_cls, test_name, wrapped_test_fn)
        return test_cls
    return decorator


def multi_threaded(
    num_workers: int,
    num_runs: int = 5,
    skip_tests: Optional[list[str]] = None,
    test_prefix: str = "_original_test",
):
    """Decorator that runs a test in a multi-threaded environment."""
    def decorator(test_cls):
        for name, test_fn in test_cls.__dict__.copy().items():
            if not (name.startswith(test_prefix) and callable(test_fn)):
                continue

            name = f"test{name[len(test_prefix):]}"
            if skip_tests is not None:
                if any(test_name in name for test_name in skip_tests):
                    continue

            def multi_threaded_test_fn(self, capfd, *args, __test_fn__=test_fn, **kwargs):
                barrier = threading.Barrier(num_workers)

                def closure():
                    barrier.wait()
                    for _ in range(num_runs):
                        __test_fn__(self, *args, **kwargs)

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_workers
                ) as executor:
                    futures = []
                    for _ in range(num_workers):
                        futures.append(executor.submit(closure))
                    # We should call future.result() to re-raise an exception if test has
                    # failed
                    assert len(list(f.result() for f in futures)) == num_workers

                gc.collect()
                assert Context._get_live_count() == 0

                captured = capfd.readouterr()
                if len(captured.err) > 0:
                    if "ThreadSanitizer" in captured.err:
                        raise RuntimeError(f"ThreadSanitizer reported warnings:\n{captured.err}")
                    else:
                        pass
                        # There are tests that write to stderr, we should ignore them
                        # raise RuntimeError(f"Other error:\n{captured.err}")

            setattr(test_cls, f"{name}_multi_threaded", multi_threaded_test_fn)

        return test_cls
    return decorator


@multi_threaded(num_workers=6, num_runs=20)
@add_existing_tests(test_prefix="_original_test")
class TestAllMultiThreaded:
    @pytest.fixture(scope='class')
    def teardown(self):
        if hasattr(self, "output_folder"):
            self.output_folder.cleanup()

    def _original_test_create_context(self):
        with Context() as ctx:
            print(ctx._get_live_count())
            print(ctx._get_live_module_count())
            print(ctx._get_live_operation_count())
            print(ctx._get_live_operation_objects())
            print(ctx._get_context_again() is ctx)
            print(ctx._clear_live_operations())

    def _original_test_create_module_with_consts(self):
        py_values = [123, 234, 345]
        with Context() as ctx:
            module = Module.create(loc=Location.file("foo.txt", 0, 0))

            dtype = IntegerType.get_signless(64)
            with InsertionPoint(module.body), Location.name("a"):
                arith.constant(dtype, py_values[0])

            with InsertionPoint(module.body), Location.name("b"):
                arith.constant(dtype, py_values[1])

            with InsertionPoint(module.body), Location.name("c"):
                arith.constant(dtype, py_values[2])
