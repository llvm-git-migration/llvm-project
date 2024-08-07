"""
Test how lldb reacts to ambiguous commands
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AmbiguousCommandTestCase(TestBase):
    @no_debug_info_test
    def test_ambiguous_command(self):
        command_interpreter = self.dbg.GetCommandInterpreter()
        self.assertTrue(command_interpreter, VALID_COMMAND_INTERPRETER)
        result = lldb.SBCommandReturnObject()

        command_interpreter.HandleCommand("scr 1+1", result)
        self.assertTrue(result.Succeeded())
