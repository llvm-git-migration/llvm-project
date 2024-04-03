# Compiles an OpenCL C - or assembles an LL file - to bytecode
#
# Arguments:
# * TRIPLE <string>
#     Target triple for which to compile the bytecode file.
# * INPUT <string>
#     File to compile/assemble to bytecode
# * OUTPUT <string>
#     Bytecode file to generate
# * EXTRA_OPTS <string> ...
#     List of compiler options to use. Note that some are added by default.
# * DEPENDENCIES <string> ...
#     List of extra dependencies to inject
#
# Depends on the libclc::clang and libclc::llvm-as targets for compiling and
# assembling, respectively.
function(compile_to_bc)
  cmake_parse_arguments(ARG
    ""
    "TRIPLE;INPUT;OUTPUT"
    "EXTRA_OPTS;DEPENDENCIES"
    ${ARGN}
  )

  # If this is an LLVM IR file (identified soley by its file suffix),
  # pre-process it with clang to a temp file, then assemble that to bytecode.
  set( TMP_SUFFIX )
  get_filename_component( FILE_EXT ${ARG_INPUT} EXT )
  if( NOT ${FILE_EXT} STREQUAL ".ll" )
    # Pass '-c' when not running the preprocessor
    set( PP_OPTS -c )
  else()
    set( PP_OPTS -E;-P )
    set( TMP_SUFFIX .tmp )
  endif()

  set( TARGET_ARG )
  if( ARG_TRIPLE )
    set( TARGET_ARG "-target" ${ARG_TRIPLE} )
  endif()

  add_custom_command(
    OUTPUT ${ARG_OUTPUT}${TMP_SUFFIX}
    COMMAND libclc::clang
      ${TARGET_ARG}
      ${PP_OPTS}
      ${ARG_EXTRA_OPTS}
      -MD -MF ${ARG_OUTPUT}.d -MT ${ARG_OUTPUT}${TMP_SUFFIX}
      # LLVM 13 enables standard includes by default - we don't want
      # those when pre-processing IR. We disable it unconditionally.
      $<$<VERSION_GREATER_EQUAL:${LLVM_PACKAGE_VERSION},13.0.0>:-cl-no-stdinc>
      -emit-llvm
      -o ${ARG_OUTPUT}${TMP_SUFFIX}
      -x cl
      ${ARG_INPUT}
    DEPENDS
      libclc::clang
      ${ARG_INPUT}
      ${ARG_DEPENDENCIES}
    DEPFILE ${ARG_OUTPUT}.d
  )

  if( ${FILE_EXT} STREQUAL ".ll" )
    add_custom_command(
      OUTPUT ${ARG_OUTPUT}
      COMMAND libclc::llvm-as -o ${ARG_OUTPUT} ${ARG_OUTPUT}${TMP_SUFFIX}
      DEPENDS libclc::llvm-as ${ARG_OUTPUT}${TMP_SUFFIX}
    )
  endif()
endfunction()

# Links together one or more bytecode files
#
# Arguments:
# * OUTPUT <string>
#     Bytecode file to generate
# * INPUT <string> ...
#     List of bytecode files to link together
function(link_bc)
  cmake_parse_arguments(ARG
    ""
    "OUTPUT"
    "INPUTS"
    ${ARGN}
  )

  add_custom_command(
    OUTPUT ${ARG_OUTPUT}
    COMMAND libclc::llvm-link -o ${ARG_OUTPUT} ${ARG_INPUTS}
    DEPENDS libclc::llvm-link ${ARG_INPUTS}
  )
endfunction()
