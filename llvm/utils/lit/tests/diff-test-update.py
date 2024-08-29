# RUN: not %{lit} --update-tests -v %S/Inputs/diff-test-update | FileCheck %s

# CHECK: # update-diff-test: could not deduce source and target from {{.*}}/Inputs/diff-test-update/1.in and {{.*}}/Inputs/diff-test-update/2.in
# CHECK: # update-diff-test: copied {{.*}}/Inputs/diff-test-update/my-file.txt to {{.*}}/Inputs/diff-test-update/my-file.expected
# CHECK: # update-diff-test: copied {{.*}}/Inputs/diff-test-update/Output/diff-tmp.test.tmp.txt to {{.*}}/Inputs/diff-test-update/diff-t-out.txt
