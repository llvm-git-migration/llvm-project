<!--===- docs/DoConcurrentMappingToOpenMP.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# `DO CONCURENT` mapping to OpenMP

```{contents}
---
local:
---
```

This document seeks to describe the effort to parallelize `do concurrent` loops
by mapping them to OpenMP worksharing constructs. The goals of this document
are:
* Describing how to instruct `flang` to map `DO CONCURENT` loops to OpenMP
  constructs.
* Tracking the current status of such mapping.
* Describing the limitations of the current implmenentation.
* Describing next steps.
* Tracking the current upstreaming status (from the AMD ROCm fork).

## Usage

In order to enable `do concurrent` to OpenMP mapping, `flang` adds a new
compiler flag: `-fdo-concurrent-parallel`. This flags has 3 possible values:
1. `host`: this maps `do concurent` loops to run in parallel on the host CPU.
   This maps such loops to the equivalent of `omp parallel do`.
2. `device`: this maps `do concurent` loops to run in parallel on a device
   (GPU). This maps such loops to the equivalent of `omp target teams
   distribute parallel do`.
3. `none`: this disables `do concurrent` mapping altogether. In such case, such
   loops are emitted as sequential loops.

The above compiler switch is currently avaialble only when OpenMP is also
enabled. So you need to provide the following options to flang in order to
enable it:
```
flang ... -fopenmp -fdo-concurrent-parallel=[host|device|none] ...
```

## Current status

Under the hood, `do concurrent` mapping is implemented in the
`DoConcurrentConversionPass`. This is still an experimental pass which means
that:
* It has been tested in a very limited way so far.
* It has been tested mostly on simple synthetic inputs.

To describe current status in more detail, following is a description of how
the pass currently behaves for single-range loops and then for multi-range
loops. The following sub-sections describe the status of the downstream 
implementation on the AMD's ROCm fork(*). We are working on upstreaming the
downstream implementation gradually and this document will be updated to reflect
such upstreaming process. Example LIT tests referenced below might also be only
be available in the ROCm fork and will upstream with the relevant parts of the
code.

(*) https://github.com/ROCm/llvm-project/blob/amd-staging/flang/lib/Optimizer/OpenMP/DoConcurrentConversion.cpp

### Single-range loops

Given the following loop:
```fortran
  do concurrent(i=1:n)
    a(i) = i * i
  end do
```

#### Mapping to `host`

Mapping this loop to the `host`, generates MLIR operations of the following
structure:

```mlir
%4 = fir.address_of(@_QFEa) ...
%6:2 = hlfir.declare %4 ...

omp.parallel {
  // Allocate private copy for `i`.
  // TODO Use delayed privatization.
  %19 = fir.alloca i32 {bindc_name = "i"}
  %20:2 = hlfir.declare %19 {uniq_name = "_QFEi"} ...

  omp.wsloop {
    omp.loop_nest (%arg0) : index = (%21) to (%22) inclusive step (%c1_2) {
      %23 = fir.convert %arg0 : (index) -> i32
      // Use the privatized version of `i`.
      fir.store %23 to %20#1 : !fir.ref<i32>
      ...

      // Use "shared" SSA value of `a`.
      %42 = hlfir.designate %6#0
      hlfir.assign %35 to %42
      ...
      omp.yield
    }
    omp.terminator
  }
  omp.terminator
}
```

#### Mapping to `device`

Mapping the same loop to the `device`, generates MLIR operations of the
following structure:

```mlir
// Map `a` to the `target` region. The pass automatically detects memory blocks
// and maps them to device. Currently detection logic is still limited and a lot
// of work is going into making it more capable.
%29 = omp.map.info ... {name = "_QFEa"}
omp.target ... map_entries(..., %29 -> %arg4 ...) {
  ...
  %51:2 = hlfir.declare %arg4
  ...
  omp.teams {
    // Allocate private copy for `i`.
    // TODO Use delayed privatization.
    %52 = fir.alloca i32 {bindc_name = "i"}
    %53:2 = hlfir.declare %52
    ...

    omp.parallel {
      omp.distribute {
        omp.wsloop {
          omp.loop_nest (%arg5) : index = (%54) to (%55) inclusive step (%c1_9) {
            // Use the privatized version of `i`.
            %56 = fir.convert %arg5 : (index) -> i32
            fir.store %56 to %53#1
            ...
            // Use the mapped version of `a`.
            ... = hlfir.designate %51#0
            ...
          }
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  omp.terminator
}
```

### Multi-range loops

The pass currently supports multi-range loops as well. Given the following
example:

```fortran
   do concurrent(i=1:n, j=1:m)
       a(i,j) = i * j
   end do
```

The generated `omp.loop_nest` operation look like:

```mlir
omp.loop_nest (%arg0, %arg1)
    : index = (%17, %19) to (%18, %20)
    inclusive step (%c1_2, %c1_4) {
  fir.store %arg0 to %private_i#1 : !fir.ref<i32>
  fir.store %arg1 to %private_j#1 : !fir.ref<i32>
  ...
  omp.yield
}
```

It is worth noting that we have privatized versions for both iteration
variables: `i` and `j`. These are locally allocated inside the parallel/target
OpenMP region similar to what the single-range example in previous section
shows.

#### Multi-range and perfectly-nested loops

Currently, on the `FIR` dialect level, the following loop:
```fortran
do concurrent(i=1:n, j=1:m)
  a(i,j) = i * j
end do
```
is modelled as a nest of `fir.do_loop` ops such that the outer loop's region
contains:
  1. The operations needed to assign/update the outer loop's induction variable.
  1. The inner loop itself.

So the MLIR structure looks similar to the following:
```mlir
fir.do_loop %arg0 = %11 to %12 step %c1 unordered {
  ...
  fir.do_loop %arg1 = %14 to %15 step %c1_1 unordered {
    ...
  }
}
```
This applies to multi-range loops in general; they are represented in the IR as
a nest of `fir.do_loop` ops with the above nesting structure.

Therefore, the pass detects such "perfectly" nested loop ops to identify multi-range
loops and map them as "collapsed" loops in OpenMP.

#### Further info regarding loop nest detection

Loop-nest detection is currently limited to the scenario described in the previous
section. However, this is quite limited and can be extended in the future to cover
more cases. For example, for the following loop nest, even thought, both loops are
perfectly nested; at the moment, only the outer loop is parallized:
```fortran
do concurrent(i=1:n)
  do concurrent(j=1:m)
    a(i,j) = i * j
  end do
end do
```

Similary for the following loop nest, even though the intervening statement `x = 41`
does not have any memory effects that would affect parallization, this nest is
not parallized as well (only the outer loop is).

```fortran
do concurrent(i=1:n)
  x = 41
  do concurrent(j=1:m)
    a(i,j) = i * j
  end do
end do
```

The above also has the consequence that the `j` variable will **not** be
privatized in the OpenMP parallel/target region. In other words, it will be
treated as if it was a `shared` variable. For more details about privatization,
see the "Data environment" section below.

See `flang/test/Transforms/DoConcurrent/loop_nest_test.f90` for more examples
of what is and is not detected as a perfect loop nest.

### Data environment

By default, variables that are used inside a `do concurernt` loop nest are
either treated as `shared` in case of mapping to `host`, or mapped into the
`target` region using a `map` clause in case of mapping to `device`. The only
exceptions to this are:
  1. the loop's iteration variable(s) (IV) of **perfect** loop nests. In that
     case, for each IV, we allocate a local copy as shown the by the mapping
     examples above.
  1. any values that are from allocations outside the loop nest and used
     exclusively inside of it. In such cases, a local privatized
     value is created in the OpenMP region to prevent multiple teams of threads
     from accessing and destroying the same memory block which causes runtime
     issues. For an example of such cases, see
     `flang/test/Transforms/DoConcurrent/locally_destroyed_temp.f90`.

Implicit mapping detection (for mapping to the GPU) is still quite limited and
work to make it smarter is underway for both OpenMP in general and `do concurrent`
mapping.

#### Non-perfectly-nested loops' IVs

For non-perfectly-nested loops, the IVs are still treated as `shared` or
`map` entries as pointed out above. This **might not** be consistent with what
the Fortran specficiation tells us. In particular, taking the following
snippets from the spec (version 2023) into account:

> § 3.35
> ------
> construct entity
> entity whose identifier has the scope of a construct

> § 19.4
> ------
>  A variable that appears as an index-name in a FORALL or DO CONCURRENT
>  construct, or ... is a construct entity. A variable that has LOCAL or
>  LOCAL_INIT locality in a DO CONCURRENT construct is a construct entity.
> ...
> The name of a variable that appears as an index-name in a DO CONCURRENT
> construct, FORALL statement, or FORALL construct has a scope of the statement
> or construct. A variable that has LOCAL or LOCAL_INIT locality in a DO
> CONCURRENT construct has the scope of that construct.

From the above quotes, it seems there is an equivalence between the IV of a `do
concurrent` loop and a variable with a `LOCAL` locality specifier (equivalent
to OpenMP's `private` clause). Which means that we should probably
localize/privatize a `do concurernt` loop's IV even if it is not perfectly
nested in the nest we are parallelizing. For now, however, we **do not** do
that as pointed out previously. In the near future, we propose a middle-ground
solution (see the Next steps section for more details).

## Next steps

### Delayed privatization

So far, we emit the privatization logic for IVs inline in the parallel/target
region. This is enough for our purposes right now since we don't
localize/privatize any sophisticated types of variables yet. Once we have need
for more advanced localization through `do concurrent`'s locality specifiers
(see below), delayed privatization will enable us to have a much cleaner IR.
Once delayed privatization's implementation upstream is supported for the
required constructs by the pass, we will move to it rather than inlined/early
privatization.

### Locality specifiers for `do concurrent`

Locality specifiers will enable the user to control the data environment of the
loop nest in a more fine-grained way. Implementing these specifiers on the
`FIR` dialect level is needed in order to support this in the
`DoConcurrentConversionPass`.

Such specifiers will also unlock a potential solution to the
non-perfectly-nested loops' IVs issue described above. In particular, for a
non-perfectly nested loop, one middle-ground proposal/solution would be to:
* Emit the loop's IV as shared/mapped just like we do currently.
* Emit a warning that the IV of the loop is emitted as shared/mapped.
* Given support for `LOCAL`, we can recommend the user to explicitly
  localize/privatize the loop's IV if they choose to.

#### Sharing TableGen clause records from the OpenMP dialect

At the moment, the FIR dialect does not have a way to model locality specifiers
on the IR level. Instead, something similar to early/eager privatization in OpenMP
is done for the locality specifiers in `fir.do_loop` ops. Having locality specifier
modelled in a way similar to delayed privatization (i.e. the `omp.private` op) and
reductions (i.e. the `omp.delcare_reduction` op) can make mapping `do concurrent`
to OpenMP (and other parallization models) much easier.

Therefore, one way to approach this problem is to extract the TableGen records
for relevant OpenMP clauses in a shared dialect for "data environment management"
and use these shared records for OpenMP, `do concurrent`, and possibly OpenACC
as well.

### More advanced detection of loop nests

As pointed out earlier, any intervening code between the headers of 2 nested
`do concurrent` loops prevents us currently from detecting this as a loop nest.
In some cases this is overly conservative. Therefore, a more flexible detection
logic of loop nests needs to be implemented.

### Data-dependence analysis

Right now, we map loop nests without analysing whether such mapping is safe to
do or not. We probalby need to at least warn the use of unsafe loop nests due
to loop-carried dependencies.

### Non-rectangular loop nests

So far, we did not need to use the pass for non-rectangular loop nests. For
example:
```fortran
do concurrent(i=1:n)
  do concurrent(j=i:n)
    ...
  end do
end do
```
We defer this to the (hopefully) near future when we get the conversion in a
good share for the samples/projects at hand.

### Generalizing the pass to other parallization models

Once we have a stable and capable `do concurrent` to OpenMP mapping, we can take
this in a more generalized direction and allow the pass to target other models;
e.g. OpenACC. This goal should be kept in mind from the get-go even while only
targeting OpenMP.


## Upstreaming status

- [x] Command line options for `flang` and `bbc`.
- [x] Conversion pass skeleton (no transormations happen yet).
- [x] Status description and tracking document (this document).
- [ ] Basic host/CPU mapping support.
- [ ] Basic device/GPU mapping support.
- [ ] More advanced host and device support (expaned to multiple items as needed).
