============================
LLVM |release| Release Notes
============================

.. contents::
    :local:

.. only:: PreRelease

  .. warning::
     These are in-progress notes for the upcoming LLVM |version| release.
     Release notes for previous releases can be found on
     `the Download Page <https://releases.llvm.org/download.html>`_.


Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release |release|.  Here we describe the status of LLVM, including major improvements
from the previous release, improvements in various subprojects of LLVM, and
some of the current users of the code.  All LLVM releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

For more information about LLVM, including information about the latest
release, please check out the `main LLVM web site <https://llvm.org/>`_.  If you
have questions or comments, the `Discourse forums
<https://discourse.llvm.org>`_ is a good place to ask
them.

Note that if you are reading this file from a Git checkout or the main
LLVM web page, this document applies to the *next* release, not the current
one.  To see the release notes for a specific release, please see the `releases
page <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================
.. NOTE
   For small 1-3 sentence descriptions, just add an entry at the end of
   this list. If your description won't fit comfortably in one bullet
   point (e.g. maybe you would like to give an example of the
   functionality, or simply have a lot to talk about), see the `NOTE` below
   for adding a new subsection.

* ...

Update on required toolchains to build LLVM
-------------------------------------------

Changes to the LLVM IR
----------------------

Changes to LLVM infrastructure
------------------------------

* Minimum Clang version to build LLVM in C++20 configuration has been updated to clang-17.0.6.

Changes to building LLVM
------------------------

Changes to TableGen
-------------------

Changes to Interprocedural Optimizations
----------------------------------------

Changes to the AArch64 Backend
------------------------------

Changes to the AMDGPU Backend
-----------------------------

Changes to the ARM Backend
--------------------------

<<<<<<< HEAD
* Added support for Cortex-M52 CPUs.
* Added execute-only support for Armv6-M.

=======
>>>>>>> faf555f93f3628b7b2b64162c02dd1474540532e
Changes to the AVR Backend
--------------------------

Changes to the DirectX Backend
------------------------------

Changes to the Hexagon Backend
------------------------------

Changes to the LoongArch Backend
--------------------------------

Changes to the MIPS Backend
---------------------------

Changes to the PowerPC Backend
------------------------------

Changes to the RISC-V Backend
-----------------------------

<<<<<<< HEAD
* The Zfa extension version was upgraded to 1.0 and is no longer experimental.
* Zihintntl extension version was upgraded to 1.0 and is no longer experimental.
* Intrinsics were added for Zk*, Zbb, and Zbc. See https://github.com/riscv-non-isa/riscv-c-api-doc/blob/master/riscv-c-api.md#scalar-bit-manipulation-extension-intrinsics
* Default ABI with F but without D was changed to ilp32f for RV32 and to lp64f for RV64.
* The Zvbb, Zvbc, Zvkb, Zvkg, Zvkn, Zvknc, Zvkned, Zvkng, Zvknha, Zvknhb, Zvks,
  Zvksc, Zvksed, Zvksg, Zvksh, and Zvkt extension version was upgraded to 1.0
  and is no longer experimental.  However, the C intrinsics for these extensions
  are still experimental.  To use the C intrinsics for these extensions,
  ``-menable-experimental-extensions`` needs to be passed to Clang.
* XSfcie extension and SiFive CSRs and instructions that were associated with
  it have been removed. None of these CSRs and instructions were part of
  "SiFive Custom Instruction Extension" as SiFive defines it. The LLVM project
  needs to work with SiFive to define and document real extension names for
  individual CSRs and instructions.
* ``-mcpu=sifive-p450`` was added.
* CodeGen of RV32E/RV64E was supported experimentally.
* CodeGen of ilp32e/lp64e was supported experimentally.

=======
>>>>>>> faf555f93f3628b7b2b64162c02dd1474540532e
Changes to the WebAssembly Backend
----------------------------------

Changes to the Windows Target
-----------------------------

Changes to the X86 Backend
--------------------------

Changes to the OCaml bindings
-----------------------------

Changes to the Python bindings
------------------------------

Changes to the C API
--------------------

Changes to the CodeGen infrastructure
-------------------------------------

Changes to the Metadata Info
---------------------------------

Changes to the Debug Info
---------------------------------

Changes to the LLVM tools
---------------------------------

Changes to LLDB
---------------------------------

Changes to Sanitizers
---------------------

Other Changes
-------------

External Open Source Projects Using LLVM 19
===========================================

* A project...

Additional Information
======================

A wide variety of additional information is available on the `LLVM web page
<https://llvm.org/>`_, in particular in the `documentation
<https://llvm.org/docs/>`_ section.  The web page also contains versions of the
API documentation which is up-to-date with the Git version of the source
code.  You can access versions of these documents specific to this release by
going into the ``llvm/docs/`` directory in the LLVM tree.

If you have any questions or comments about LLVM, please feel free to contact
us via the `Discourse forums <https://discourse.llvm.org>`_.
