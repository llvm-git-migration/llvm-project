<!--===- docs/OpenMP-declare-target.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Introduction to Declare Target

In OpenMP `declare target` is a directive that can be applied to a function or variable (primarily global) to notate to the compiler that it should be generated in a particular devices environment. In essence whether something should be emitted for host or device, or both. An example of its usage for both data and functions can be seen below.

```Fortran
module test_0
    integer :: sp = 0
!$omp declare target link(sp)
end module test_0

program main
    use test_0
!$omp target map(tofrom:sp)
    sp = 1
!$omp end target
end program
```

In the above example, we created a variable in a seperate module, mark it as `declare target` and then map it, embedding it into the device IR and assigning to it. 


```Fortran
function func_t_device() result(i)
    !$omp declare target to(func_t_device) device_type(nohost)
        INTEGER :: I
        I = 1
end function func_t_device

program main
!$omp target
    call func_t_device()
!$omp end target
end program
```

In the above example, we are stating that a function is required on device utilising `declare target`, and that we will not be utilising it on host, so we are in theory free to remove or ignore it. A user could also in this case, leave off the `declare target` from the function and it would be implicitly marked `declare target any` (for both host and device), as it's been utilised within a target region.

# Declare Target as represented in the OpenMP Dialect

In the OpenMP Dialect `declare target` is not represented by a specific `operation` instead it's a OpenMP dialect specific `attribute` that can be applied to any operation in any dialect. This helps to simplify the utilisation of it, instead of replacing or modifying existing global or function `operations` in a dialect it applies to it as extra metadata that the lowering can use in different ways as is neccesary. 

The `attribute` is composed of multiple fields representing the clauses you would find on the `declare target` directive i.e. device type (`nohost`, `any`, `host`) or the capture clause (`link` or `to`). A small example of `declare target` applied to an Fortran `real` can be found below:

```MLIR
fir.global internal @_QFEi {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : f32 {
    %0 = fir.undefined f32
    fir.has_value %0 : f32
}
```

This would look similar for function style `operations`.

The application and access of this attribute is aided by an OpenMP Dialect MLIR Interface named `DeclareTargetInterface`, which can be utilised on operations to access the appropriate interface functions, e.g.:

```C++
auto declareTargetGlobal = llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(Op.getOperation());
declareTargetGlobal.isDeclareTarget();
```

# Declare Target Fortran OpenMP Lowering

The initial lowering of `declare target` to MLIR for both use-cases is done inside of the usual OpenMP lowering in flang/lib/Lower/OpenMP.cpp, however some direct calls to `declare target` related functions from Flang's Bridge in flang/lib/Lower/Bridge.cpp are made.

The marking of operations with the declare target attribute happens in two phases, the second one optional contingent on the first failing to apply the attribute due to the operation not being generated yet, the main notable case this occurs currently is when a Fortran function interface has been marked. 

The initial phase happens when the declare target directive and it's clauses are initially processed, with the primary data gathering for the directive and clause happening in a function called `getDeclareTargetInfo` which is then used to feed the `markDeclareTarget` function which does the actual marking utilising the `DeclareTargetInterface`, if it encounters something that has been marked twice over multiple directives with two differing device types (e.g. `host`, `nohost`), then it will swap the device type to `any`.

Whenever we invoke `genFIR` on an `OpenMPDeclarativeConstruct` from Bridge, we are also invoking another function 
called `gatherOpenMPDeferredDeclareTargets` which gathers information relevant to the application of the `declare target` attribute (symbol that it should be applied to, device type clause, and capture clause) when processing `declare target` declarations, storing the data in a vector that is part of Bridge's instantiation of the `AbstractConverter`. This data is only stored if we encounter a function or variable symbol that does not have an operation instantiated for it yet, unfortunately this cannot happen as part of the initial marking as we must store this data in Bridge and only have access to the abstract version of the converter via the OpenMP lowering. 

This information is used in the second phase, which is a form of deferred processing of the `declare target` marked operations that have delayed generation, this is done via the function `markOpenMPDeferredDeclareTargetFunctions` which is called from Bridge at the end of the lowering process allowing us to mark those where possible. It iterates over the data gathered by `gatherOpenMPDeferredDeclareTargets` checking if any of the recorded symbols have now had their corresponding operations instantiated and applying where possible utilising `markDeclareTarget`.
However, it must be noted that it is still possible for operations not to be generated for certain symbols, in particular the case of function interfaces that are not directly used or defined within the current module, this means we cannot emit errors in the case of left-over unmarked symbols, these must (and should) be caught by the initial semantic analysis.

NOTE: `declare target` can be applied to implicit `SAVE` attributed variables, however, by default Flang does not represent these as `GlobalOp`'s which means we cannot tag and lower them as `declare target` normally, instead similarly to the way `threadprivate` handles these cases, we raise and initialize the variable as an internal `GlobalOp` and apply the attribute. This occurs in the flang/lib/Lower/OpenMP.cpp function `genDeclareTargetIntGlobal`.

# Declare Target Transformation Passes for Flang

There are currently two passes within Flang that are related to the processing of `declare target`:
* `OMPMarkDeclareTarget` - This pass is in charge of marking functions captured (called from) in `target` regions or other `declare target` marked functions as `declare target`, it does so recursively, e.g. nested calls will also be implicitly marked. It currently will try to mark things as conservatively as possible, i.e. if captured in a `target` region it will apply `nohost`, unless it encounters something with `host` in which case it will apply the any device type (if it's already `any`, then it's left untouched). Functions are handled similarly, except we utilise the parents device type where possible.   
* `OMPFunctionFiltering` - This is executed after `OMPMarkDeclareTarget`, and currently only for device, its job is to conservatively remove functions from the module where possible. This helps make sure incompatible code from the host is not lowered for device (although, a user can still self inject incompatible code, but this mechanism allows them to avoid that). Functions with `target` regions in them are preserved as they may be neccesary to maintain (e.g. reverse offloading in the future), otherwise, we will remove any function marked as a `declare target host` function and any uses will be replaced with `undef`'s so that other passes can appropriately clean them up and in the meantime we don't break verification.

While this infrastructure is generally applicable to more than just Flang, we currently only utilise them in the Flang frontend and they are part of the Flang codebase, rather than the OpenMP dialect codebase. 

# Declare Target OpenMP Dialect To LLVM-IR Lowering

The OpenMP dialect lowering of `declare target` is a little unique currently, as it's not an `operation` and is an `attribute` we process it utilising the LLVM Target lowerings `amendOperation`, which occurs immediately after an operation has been lowered to LLVM-IR. As it can be applicable to multiple different operations, we must
specialise this function for each operation type that we may encounter. Currently this is `GlobalOp`'s and 
`FuncOp`'s.

In the case where we encounter a `FuncOp` our processing is fairly simple, if we're processing the device code, we will finish up our removal of `host` marked functions, anything we could not remove previously we now remove, e.g. if it had a `target` directive in it (which we need to keep a hold of to this point, to actually outline the `target` kernel for device). This hopefully leaves us with only `any`, `device` or undeterminable functions left in the module to lower further, reducing the possibiltiy of device incompatible code being in the module.

For `GlobalOp`'s, the processing is a little more complex, we currently leverage two OMPIRBuilder functions which we have inherited from Clang and moved to the `OMPIRBuilder` to share across the two compiler frontends `registerTargetGlobalVariable` and `getAddrOfDeclareTargetVar`. These two functions are actually recursive and invoke each other depending on the clauses and options provided to the `OMPIRBuilder` (in particular unified shared memory), but the main functionality they provide is the generation of a new global pointer for device with a "ref_" prefix, and enqueuing metadata generation by the `OMPIRBuilder` at the end of the module, for both host and device that links the newly generated device global pointer and the host pointer together across the two modules (and resulting binaries). 

Two things of note about the `GlobalOp` processing, the first is that similarly to other metadata (e.g. for `TargetOp`) that is shared across both host and device modules, the device needs access to the previously generated host IR file, which is done through another `attribute` applied to the `ModuleOp` by the compiler frontend. The file is loaded in and consumed by the `OMPIRBuilder` to populate it's `OffloadInfoManager` data structures, keeping host and device appropriately synchronised.

The second (and more important to remember) is that as we effectively replace the original LLVM-IR generated for the `declare target` marked `GlobalOp` we have some corrections we need to do for `TargetOp`'s (or other region operations that use them directly) which still refer to the original lowered global operation. This is done via `handleDeclareTargetMapVar` which is invoked as the final function and alteration to the lowered `target` region, it's only invoked for device as it's only required in the case where we have emitted the "ref" pointer , and it effectively replaces all uses of the originally lowered global symbol, with our new global ref pointer's symbol. Currently we do not remove or delete the old symbol, this is due to the fact that the same symbol can be utilised across multiple target regions, if we remove it, we risk breaking lowerings of target regions that will be processed at a later time. To appropriately delete these no longer neccesary symbols we would need a deferred removal process at the end of the module, which is currently not in place. It may be possible to store this information in the OMPIRBuilder and then perform this cleanup process on finalization, but this is open for discussion and implementation still.

# Current Support

For the moment, `declare target` should work for:
* Marking functions/subroutines and function/subroutine interfaces for generation on host, device or both.
* Implicit function/subroutine capture for calls emitted in a `target` region or explicitly marked `declare 
   target` function/subroutine. Note: Calls made via arguments passed to other functions must still be 
   themselves marked `declare target`, e.g. passing a `C` function pointer and invoking it, then the interface
   and the `C` function in the other module must be marked `declare target`, with the same type of 
   marking as indicated by the specification.
* Marking global variables with `declare target`'s `link` clause and mapping the data to the device data 
   environment utilising `declare target` (may not work for all types yet, but for scalars and arrays
   of scalars, it should).

Doesn't work for, or needs further testing for:
* Marking the following types with `declare target link` (needs further testing):
    * Descriptor based types, e.g. pointers/allocatables.
    * Derived types.
    * Members of derived types (use-case needs legality checking with OpenMP specification).
* Marking global variables with `declare target`'s `to` clause, a lot of the lowering should exist, but it needs further testing and likely some further changes to fully function.
