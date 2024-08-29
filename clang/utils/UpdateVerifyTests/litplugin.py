import sys
from lit.formats import ShTest
from UpdateVerifyTests.core import check_expectations
import re

verify_r = re.compile(r"-verify(?:=(\w+))?")

def get_verify_prefixes(command):
    def get_default(prefix):
        if prefix:
            return prefix
        return "expected"

    prefixes = set()
    for arg in command.args:
        m = verify_r.match(arg)
        if not m:
            continue
        prefix = m[1]
        if not prefix:
            prefix = "expected"
        prefixes.add(prefix)
    return prefixes

def verify_test_updater(result):
    if not result.stderr:
        return None
    prefixes = get_verify_prefixes(result.command)
    if not prefixes:
        return None
    if len(prefixes) > 1:
        return f"update-verify-test: not updating because of multiple prefixes - {prefixes}"
    [prefix] = prefixes
    return check_expectations(result.stderr.splitlines(), prefix)

