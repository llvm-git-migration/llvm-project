# InstCombine contributor guide

This guide lays out a series of rules that contributions to InstCombine should
follow. **Following these rules will results in much faster PR approvals.**

## Tests

### Precommit tests

Tests for new optimizations or miscompilation fixes should be pre-committed.
This means that you first commit the test with CHECK lines prior to your fix.
Your actual change will then only contain CHECK line diffs relative to that
baseline.

This means that pull requests should generally contain two commits: First,
one commit adding new tests with baseline check lines. Second, a commit with
functional changes and test diffs.

If the second commit in your PR does not contain test diffs, you did something
wrong. Either you made a mistake when generating CHECK lines, or your tests are
not actually affected by your patch.

Exceptions: When fixing assertion failures or infinite loops, do not pre-commit
tests.

### Use `update_test_checks.py`

CHECK lines should be generated using the `update_test_checks.py` script. Do
**not** manually edit check lines after using it.

Be sure to use the correct opt binary when using the script. For example, if
your build directory is `build`, then you'll want to run:

```sh
llvm/utils/update_test_checks.py --opt-binary build/bin/opt \
    llvm/test/Transforms/InstCombine/the_test.ll
```

Exceptions: Hand-written CHECK lines are allowed for debuginfo tests.

### General testing considerations

Place all tests relating to a transform into a single file. If you are adding
a regression test for a crash/miscompile in an existing transform, find the
file where the existing tests are located. A good way to do that is to comment
out the transform and see which tests fail.

Make tests minimal. Only test exactly the pattern being transformed. If your
original motivating case is a larger pattern that your fold enables to
optimize in some non-trivial way, you may add it as well -- however, the bulk
of the test coverage should be minimal.

Give tests short, but meaningful names. Don't call them `@test1`, `@test2` etc. 
For example, a test checking multi-use behavior of a fold involving the
addition of two selects might be called `@add_of_selects_multi_use`.

Add representative tests for each test category (discussed below), but don't
test all combinations of everything. If you have multi-use tests, and you have
commuted tests, you shouldn't also add commuted multi-use tests.

Prefer to keep bit-widths for tests low to improve performance of proof checking using alive2. Using `i8` is better than `i128` where possible. 

### Add negative tests

Make sure to add tests for which your transform does **not** apply. Start with
one of the test cases that succeeds and then create a sequence of negative
tests, such that **exactly one** different pre-condition of your transform is
not satisfied in each test.

### Add multi-use tests

Add multi-use tests that ensures your transform does not increase instruction
count if some instructions have additional uses. The standard pattern is to
introduce extra uses with function calls:

```llvm
declare void @use(i8)

define i8 @add_mul_const_multi_use(i8 %x) {
  %add = add i8 %x, 1
  call void @use(i8 %add)
  %mul = mul i8 %add, 3
  ret i8 %mul
}
```

Exceptions: For transform that only produce one instruction, multi-use tests
may be omitted.

### Add commuted tests

If the transform involves commutative operations, add tests with commuted
(swapped) operands.

Make sure that that the operand order stays intact in the CHECK lines of your
pre-commited tests. You should not see something like this:

```llvm
; CHECK-NEXT: [[OR:%.*]] = or i8 [[X]], [[Y]]
; ...
%or = or i8 %y, %x
```

If this happens, you may need to change one of the operands to have higher
complexity (include the "thwart" comment in that case):

```llvm
%y2 = mul i8 %y, %y ; thwart complexity-based canonicalization
%or = or i8 %y, %x
```

### Add vector tests

When possible, it is recommended to add at least one test that uses vectors
instead of scalars.

For patterns that include constants, we distinguish three kinds of tests.
The first are "splat" vectors, where all the vector elements are the same.
These tests *should* usually fold without additional effort.

```llvm
define <2 x i8> @add_mul_const_vec_splat(<2 x i8> %x) {
  %add = add <2 x i8> %x, <i8 1, i8 1>
  %mul = mul <2 x i8> %add, <i8 3, i8 3>
  ret <2 x i8> %mul
}
```

A minor variant is to replace some of the splat elements with poison. These
will often also fold without additional effort.

```llvm
define <2 x i8> @add_mul_const_vec_splat_poison(<2 x i8> %x) {
  %add = add <2 x i8> %x, <i8 1, i8 poison>
  %mul = mul <2 x i8> %add, <i8 3, i8 poison>
  ret <2 x i8> %mul
}
```

Finally, you can have non-splat vectors, where the vector elements are not
the same:

```llvm
define <2 x i8> @add_mul_const_vec_non_splat(<2 x i8> %x) {
  %add = add <2 x i8> %x, <i8 1, i8 5>
  %mul = mul <2 x i8> %add, <i8 3, i8 6>
  ret <2 x i8> %mul
}
```

Non-splat vectors will often not fold by default. You should **not** try to
make them fold, unless doing so does not add **any** additional complexity.
You should still add the test though, even if it does not fold.

### Poison flag tests

If your transform involves instructions that can have poison-generating flags,
such as `nuw` and `nsw` on `add`, you should test how these interact with the
transform.

If your transform *requires* a certain flag for correctness, make sure to add
negative tests missing the required flag.

If your transform doesn't require flags for correctness, you should have tests
for preservation behavior. If the input instructions have certain flags, are
they preserved in the output instructions, if it is valid to preserve them?
(This depends on the transform. Check with alive2.)

### Other tests

The test categories mentioned above are non-exhaustive. There may be more tests
to be added, depending on the instructions involved in the transform. Some
examples:

 * For folds involving memory accesses like load/store, check that scalable vectors and non-byte-size types (like i3) are handled correctly. Also check that volatile/atomic are handled.
 * For folds that interact with the bitwidth in some non-trivial way, check an illegal type like i13. Also confirm that the transform is correct for i1.
 * For folds that involve phis, you may want to check that the case of multiple incoming values from one block is handled correctly.

## Proofs

Your pull request description should contain one or more
[alive2 proofs](https://alive2.llvm.org/ce/) for the correctness of the
proposed transform.

### Basics

Proofs are written using LLVM IR, by specifying a `@src` and `@tgt` function.
It is possible to include multiple proofs in a single file by giving the src
and tgt functions matching suffixes.

For example, here is a pair of proofs that both `(x-y)+y` and `(x+y)-y` can
be simplified to `x` ([online](https://alive2.llvm.org/ce/z/MsPPGz)):

```llvm
define i8 @src_add_sub(i8 %x, i8 %y) {
  %add = add i8 %x, %y
  %sub = sub i8 %add, %y
  ret i8 %sub
}

define i8 @tgt_add_sub(i8 %x, i8 %y) {
  ret i8 %x
}


define i8 @src_sub_add(i8 %x, i8 %y) {
  %sub = sub i8 %x, %y
  %add = add i8 %sub, %y
  ret i8 %add
}

define i8 @tgt_sub_add(i8 %x, i8 %y) {
  ret i8 %x
}
```

### Use generic values in proofs

Proofs should operate on generic values, rather than specific constants, to the degree that this is possible.

For example, this is a **bad** proof:

```llvm
; Don't do this!
define i8 @src(i8 %x) {
  %smax = call i8 @llvm.smax.i8(i8 %x, i8 5)
  %umax = call i8 @llvm.umax.i8(i8 %smax, i8 7)
  ret i8 %umax
}

define i8 @tgt(i8 %x) {
  %smax = call i8 @llvm.smax.i8(i8 %x, i8 7)
  ret i8 %smax
}
```

The correct way to write this proof is as follows
([online](https://alive2.llvm.org/ce/z/Sgfq37)):

```
define i8 @src(i8 %x, i8 %c1, i8 %c2) {
  %cond1 = icmp sgt i8 %c1, -1
  call void @llvm.assume(i1 %cond1)
  %cond2 = icmp sgt i8 %c2, -1
  call void @llvm.assume(i1 %cond2)

  %smax = call i8 @llvm.smax.i8(i8 %x, i8 %c1)
  %umax = call i8 @llvm.umax.i8(i8 %smax, i8 %c2)
  ret i8 %umax
}

define i8 @tgt(i8 %x, i8 %c1, i8 %c2) {
  %umax = call i8 @llvm.umax.i8(i8 %c1, i8 %c2)
  %smax = call i8 @llvm.smax.i8(i8 %x, i8 %umax)
  ret i8 %smax
}
```

Note that the `@llvm.assume` intrinsic is used to specify pre-conditions for
the transform. In this case, it specifies that the constants `%c1` and `%c2`
must be non-negative.

It should be emphasized that there is, in general, no expectation that the
IR in the proofs will be transformed by the implemented fold. In the above
example, the transform would only actually apply if `%c1` and `%c2` are
actually constants, but we need to use non-constants in the proof.

### Common pre-conditions

Here are some examples of common preconditions.

```llvm
; %x is non-negative:
%nonneg = icmp sgt i8 %x, -1
call void @llvm.assume(i1 %nonneg)

; %x is a power of two:
%ctpop = call i8 @llvm.ctpop.i8(i8 %x)
%pow2 = icmp eq i8 %x, 1
call void @llvm.assume(i1 %pow2)

; %x is a power of two or zero:
%ctpop = call i8 @llvm.ctpop.i8(i8 %x)
%pow2orzero = icmp ult i8 %x, 2
call void @llvm.assume(i1 %pow2orzero)

; Adding %x and %y does not overflow in a signed sense:
%wo = call { i8, i1 } @llvm.sadd.with.overflow(i8 %x, i8 %y)
%ov = extractvalue { i8, i1 } %wo, 1
%ov.not = xor i1 %ov, true
call void @llvm.assume(i1 %ov.not)
```

### Timeouts

Alive2 proofs will sometimes produce a timeout with the following message: 

```
Alive2 timed out while processing your query.
There are a few things you can try:

- remove extraneous instructions, if any

- reduce variable widths, for example to i16, i8, or i4

- add the --disable-undef-input command line flag, which
  allows Alive2 to assume that arguments to your IR are not
  undef. This is, in general, unsound: it can cause Alive2
  to miss bugs.
```

This is good advice, follow it!

Reducing the bitwidth usually helps. For floating point numbers, you can use
the `half` type for bitwidth reduction purposes. For pointers, you can reduce
the bitwidth by specifying a custom data layout:

```
; For 16-bit pointers
target datalayout = "p:16:16"
```

If reducing the bitwidth does not help, try `-disable-undef-input`. If this
still does not work, submit your proof despite the timeout.
