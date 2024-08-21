"""
Test lldb-dap stack trace response
"""


import dap_server
from lldbsuite.test.decorators import *
import os

import lldbdap_testcase
from lldbsuite.test import lldbtest, lldbutil


class TestDAP_subtleFrames(lldbdap_testcase.DAPTestCaseBase):
    def test_subtleFrames(self):
        """
        Test that internal stack frames (such as the ones used by `std::function`)
        are marked as "subtle".
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        self.set_source_breakpoints(source, [line_number(source, "BREAK HERE")])
        self.continue_to_next_stop()

        backtrace = self.get_stackFrames()[0]
        for f in backtrace:
            if "__function" in f["name"]:
                self.assertEqual(f["presentationHint"], "subtle")
