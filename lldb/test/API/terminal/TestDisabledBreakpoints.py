"""
Test that disabling breakpoints and viewing them in a list uses the correct ANSI color settings when colors are enabled and disabled.
"""
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbpexpect import PExpectTest

import re
import io

class DisabledBreakpointsTest(PExpectTest):
    @add_test_categories(["pexpect"])
    def test_disabling_breakpoints_with_color(self):
        """Test that disabling a breakpoint and viewing the breakpoints list uses the specified ANSI color prefix."""
        import pexpect
        self.child = pexpect.spawn("expect", encoding="utf-8")

        ansi_red_color_code = "\x1b[31m"

        self.launch(use_colors=True, dimensions=(100, 100))
        self.child.sendline('settings set disable-breakpoint-ansi-prefix "${ansi.fg.red}"')
        self.child.sendline('b main')
        self.child.sendline('br dis')
        self.child.sendline('br l')
        self.child.expect_exact(ansi_red_color_code + "1:")
        self.quit()
