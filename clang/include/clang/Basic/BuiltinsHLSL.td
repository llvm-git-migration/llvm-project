//===--- BuiltinsHLSL.td - HLSL Builtin function database ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsBase.td"

def HLSLWaveActiveCountBits : LangBuiltin<"HLSL_LANG"> {
  let Spellings = ["__builtin_hlsl_wave_active_count_bits"];
  let Attributes = [NoThrow, Const];
  let Prototype = "unsigned int(bool)";
}

def HLSLCreateHandle : LangBuiltin<"HLSL_LANG"> {
  let Spellings = ["__builtin_hlsl_create_handle"];
  let Attributes = [NoThrow, Const];
  let Prototype = "void*(unsigned char)";
}
