"""
Test RISC-V expressions evaluation.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestExpressions(TestBase):
    def common_setup(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

    @skipIf(archs=no_match(["rv64gc"]))
    def test_int_arg(self):
        self.common_setup()
        self.expect_expr("foo(foo(5), foo())", result_type="int", result_value="8")

    @skipIf(archs=no_match(["rv64gc"]))
    def test_double_arg(self):
        self.common_setup()
        self.expect_expr("func_with_double_arg(1, 6.5)",result_type="int", result_value="1")

    @skipIf(archs=no_match(["rv64gc"]))
    def test_ptr_arg(self):
        self.common_setup()
        self.expect_expr(
            'func_with_ptr_arg("message")', result_type="int", result_value="2"
        )

    @skipIf(archs=no_match(["rv64gc"]))
    def test_ptr_ret_val(self):
        self.common_setup()
        self.expect("expr func_with_ptr_return()", substrs=["global"])

    @skipIf(archs=no_match(["rv64gc"]))
    def test_ptr(self):
        self.common_setup()
        self.expect(
            'expr func_with_ptr("message")',
            substrs=["message"],
        )

    @skipIf(archs=no_match(["rv64gc"]))
    def test_global_ptr(self):
        self.common_setup()
        self.expect(
            "expr func_with_ptr(g_str)",
            substrs=["global"],
        )

    @skipIf(archs=no_match(["rv64gc"]))
    def test_struct_arg(self):
        self.common_setup()
        self.expect_expr("func_with_struct_arg(s)", result_type="int", result_value="110")

    @skipIf(archs=no_match(["rv64gc"]))
    def test_unsupported_struct_arg(self):
        self.common_setup()
        self.expect_expr("func_with_double_struct_arg(u)", result_type="int", result_value="400")

    @skipIf(archs=no_match(["rv64gc"]))
    def test_double_ret_val(self):
        self.common_setup()
        self.expect_expr("func_with_double_return()", result_type="double", result_value="42")

    @skipIf(archs=no_match(["rv64gc"]))
    def test_struct_return(self):
        self.common_setup()
        self.expect_expr("func_with_struct_return()", result_type="S")

    @skipIf(archs=no_match(["rv64gc"]))
    def test_struct_double_ret_val(self):
        self.common_setup()
        self.expect_expr("func_with_double_struct_return()", result_type="U")

    @skipIf(archs=no_match(["rv64gc"]))
    def test_float_arg(self):
        self.common_setup()
        self.expect_expr("func_with_float_arg(9.99, 8.88)", result_type="int", result_value="7")

    @skipIf(archs=no_match(["rv64gc"]))
    def test_float_ret_val(self):
        self.common_setup()
        self.expect_expr("func_with_float_ret_val()", result_type="float", result_value="8")

