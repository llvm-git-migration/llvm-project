.. _ModulesInLibcxx:

=================
Modules in libc++
=================

.. warning:: Modules are an experimental feature. It has additional build
             requirements and not all libc++ configurations are supported yet.

             The work is still in an early development state and not
             considered stable nor complete

This page contains information regarding C++23 module support in libc++.
There are two kinds of modules available in Clang

 * `Clang specific modules <https://clang.llvm.org/docs/Modules.html>`_
 * `C++ modules <https://clang.llvm.org/docs/StandardCPlusPlusModules.html>`_

This page mainly discusses the C++ modules. In C++20 there are also header units,
these are not part of this document.

Overview
========

The module sources are stored in ``.cppm`` files. Modules need to be available
as BMIs, which are ``.pcm`` files for Clang. BMIs are not portable, they depend
on the compiler used and its compilation flags. Therefore there needs to be a
way to distribute the ``.cppm`` files to the user and offer a way for them to
build and use the ``.pcm`` files. It is expected this will be done by build
systems in the future. To aid early adaptor and build system vendors libc++
currently ships a CMake project to aid building modules.

.. note:: This CMake file is intended to be a temporary solution and will
          be removed in the future. The timeline for the removal depends
          on the availability of build systems with proper module support.

What works
~~~~~~~~~~

 * Building BMIs
 * Running tests using the ``std`` and ``std.compat`` module
 * Using the ``std``  and ``std.compat`` module in external projects
 * The following "parts disabled" configuration options are supported

   * ``LIBCXX_ENABLE_LOCALIZATION``
   * ``LIBCXX_ENABLE_WIDE_CHARACTERS``
   * ``LIBCXX_ENABLE_THREADS``
   * ``LIBCXX_ENABLE_FILESYSTEM``
   * ``LIBCXX_ENABLE_RANDOM_DEVICE``
   * ``LIBCXX_ENABLE_UNICODE``
   * ``LIBCXX_ENABLE_EXCEPTIONS`` [#note-no-windows]_

 * A C++20 based extension

.. note::

   .. [#note-no-windows] This configuration will probably not work on Windows
                         due to hard-coded compilation flags.

Some of the current limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 * There is no official build system support, libc++ has experimental CMake support
 * Requires CMake 3.26 for C++20 support
 * Requires CMake 3.26 for C++23 support
 * Requires CMake 3.27 for C++26 support
 * Requires Ninja 1.11
 * Requires Clang 17
 * The path to the compiler may not be a symlink, ``clang-scan-deps`` does
   not handle that case properly
 * Libc++ is not tested with modules instead of headers
 * Clang supports modules using GNU extensions, but libc++ does not work using
   GNU extensions.
 * Clang:
    * Including headers after importing the ``std`` module may fail. This is
      hard to solve and there is a work-around by first including all headers
      `bug report <https://github.com/llvm/llvm-project/issues/61465>`__.

Blockers
~~~~~~~~

  * libc++

    * Currently the tests only test with modules enabled, but do not import
      modules instead of headers. When converting tests to using modules there
      are still failures. These are under investigation.

    * It has not been determined how to fully test libc++ with modules instead
      of headers.

  * Clang

    * Some concepts do not work properly
      `bug report <https://github.com/llvm/llvm-project/issues/62943>`__.


Using in external projects
==========================

Users need to be able to build their own BMI files.

.. note:: The requirements for users to build their own BMI files will remain
   true for the foreseeable future. For now this needs to be done manually.
   Once libc++'s implementation is more mature we will reach out to build
   system vendors, with the goal that building the BMI files is done by
   the build system.

Currently this requires a local build of libc++ with modules installation enabled.
Since modules are not installed by default. You can build and install the modules
to ``<install_prefix>`` with the following commands.

.. code-block:: bash

  $ git clone https://github.com/llvm/llvm-project.git --depth 1
  $ cd llvm-project
  $ mkdir build
  $ cmake -G Ninja -S runtimes -B build -DCMAKE_C_COMPILER=<path-to-compiler> -DCMAKE_CXX_COMPILER=<path-to-compiler> -DLIBCXX_INSTALL_MODULES=ON -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind"
  $ cmake --build build -- -j $(nproc)
  $ cmake --install build --prefix <install_prefix>

This is a small sample program that uses the module ``std``. It consists of a
``CMakeLists.txt``, an ``std.cmake``, and a ``main.cpp`` file.

.. code-block:: cpp

  // main.cpp
  import std; // When importing std.compat it's not needed to import std.
  import std.compat;

  int main() {
    std::println("Hello modular world");
    ::printf("Hello compat modular world\n");
  }

.. code-block:: cmake

  # CMakeLists.txt
  cmake_minimum_required(VERSION 3.26.0 FATAL_ERROR)
  project("module"
    LANGUAGES CXX
  )

  #
  # Set language version used
  #

  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED YES)
  # Libc++ doesn't support compiler extensions for modules.
  set(CMAKE_CXX_EXTENSIONS OFF)

  #
  # Enable modules in CMake
  #

  # This is required to write your own modules in your project.
  if(CMAKE_VERSION VERSION_LESS "3.28.0")
    if(CMAKE_VERSION VERSION_LESS "3.27.0")
      set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "2182bf5c-ef0d-489a-91da-49dbc3090d2a")
    else()
      set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "aa1f7df0-828a-4fcd-9afc-2dc80491aca7")
    endif()
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)
  else()
    cmake_policy(VERSION 3.28)
  endif()

  #
  # Import the modules from libc++
  #
  include(std.cmake)

  add_executable(main main.cpp)

.. code-block:: cmake

  # std.cmake
  include(FetchContent)
  FetchContent_Declare(
    std_module
    URL "file://${LIBCXX_INSTALLED_DIR}/share/libc++/v1"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM
  )

  if (NOT std_module_POPULATED)
    FetchContent_Populate(std_module)
  endif()

  #
  # Add std static library
  #

  add_library(std)

  target_sources(std
    PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
      ${std_module_SOURCE_DIR}/std.cppm
      ${std_module_SOURCE_DIR}/std.compat.cppm
  )

  #
  # Adjust project include directories
  #

  target_include_directories(std SYSTEM PUBLIC ${LIBCXX_INSTALLED_DIR}/include/c++/v1)

  #
  # Adjust project compiler flags
  #

  target_compile_options(std
    PRIVATE
      -Wno-reserved-module-identifier
      -Wno-reserved-user-defined-literal
  )

  target_compile_options(std
    PUBLIC
      -nostdinc++
  )

  #
  # Adjust project linker flags
  #

  target_link_options(std
    INTERFACE
      -nostdlib++
      -L${LIBCXX_INSTALLED_DIR}/lib
      -Wl,-rpath,${LIBCXX_INSTALLED_DIR}/lib
  )

  target_link_libraries(std
    INTERFACE
      c++
  )
  
  #
  # Link to the std modules by default
  #

  link_libraries(std)

Building this project is done with the following steps, assuming the files
``main.cpp``, ``CMakeLists.txt``, and ``std.cmake`` are copied in the current directory.

.. code-block:: bash

  $ mkdir build
  $ cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=<path-to-compiler> -DLIBCXX_INSTALLED_DIR=<install_prefix>
  $ cmake --build build
  $ ./build/main

.. warning:: You need more than clang itself to build a project using modules.
             Specifically, you will need ``clang-scan-deps``. For example, in Ubuntu, you
             need to use ``sudo ./llvm.sh 17 all`` rather than ``sudo ./llvm.sh 17`` showed
             in `LLVM Debian/Ubuntu nightly packages <https://apt.llvm.org>`__ to install
             essential components to build this project.

.. warning:: ``<path-to-compiler>`` should point point to the real binary and
             not to a symlink.

.. warning:: When using these examples in your own projects make sure the
             compilation flags are the same for the ``std`` module and your
             project. Some flags will affect the generated code, when these
             are different the module cannot be used. For example using
             ``-pthread`` in your project and not in the module will give
             errors like

             ``error: POSIX thread support was disabled in PCH file but is currently enabled``

             ``error: module file _deps/std-build/CMakeFiles/std.dir/std.pcm cannot be loaded due to a configuration mismatch with the current compilation [-Wmodule-file-config-mismatch]``

If you have questions about modules feel free to ask them in the ``#libcxx``
channel on `LLVM's Discord server <https://discord.gg/jzUbyP26tQ>`__.

If you think you've found a bug please it using the `LLVM bug tracker
<https://github.com/llvm/llvm-project/issues>`_. Please make sure the issue
you found is not one of the known bugs or limitations on this page.
