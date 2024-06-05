from typing_extensions import override
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class CommandOverrideCallback(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.line = line_number("main.c", "Hello world.")

    def test_command_override_callback(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Retrieve the associated command interpreter from our debugger.
        ci = self.dbg.GetCommandInterpreter()
        self.assertTrue(ci, VALID_COMMAND_INTERPRETER)

        override_string = "what"
        def foo(string):
            nonlocal override_string
            override_string = string
            return False

        self.assertTrue(ci.SetCommandOverrideCallback("breakpoint", foo))
