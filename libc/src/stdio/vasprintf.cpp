//===-- Implementation of vasprintf -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/vasprintf.h"

#include "src/__support/OSUtil/io.h"
#include "src/__support/arg_list.h"
#include "src/stdio/printf.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/printf_main.h"
#include "src/stdio/printf_core/writer.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h> // malloc, realloc, free

namespace LIBC_NAMESPACE {
namespace {

LIBC_INLINE int resize_overflow_hook(cpp::string_view new_str, void *target) {
  printf_core::WriteBuffer *wb =
      reinterpret_cast<printf_core::WriteBuffer *>(target);
  size_t new_size = new_str.size() + wb->buff_cur;
  // init_buff is a stack ptr, needs to call malloc first.
  char *TmpBuf =
      static_cast<char *>((wb->buff == wb->init_buff)
                              ? malloc(new_size + 1)
                              : realloc(wb->buff, new_size + 1)); // +1 for null
  if (TmpBuf == nullptr) {
    if (wb->buff != wb->init_buff) {
      free(wb->buff);
    }
    return printf_core::ALLOCATION_ERROR;
  }
  wb->buff = TmpBuf;
  inline_memcpy(wb->buff + wb->buff_cur, new_str.data(), new_str.size());
  wb->buff_cur = new_size;
  wb->buff_len = new_size;
  return printf_core::WRITE_OK;
}

} // namespace
inline constexpr size_t DEFAULT_BUFFER_SIZE = 200;

LLVM_LIBC_FUNCTION(int, vasprintf,
                   (char **__restrict ret, const char *format, va_list vlist)) {
  internal::ArgList args(vlist); // This holder class allows for easier copying
                                 // and pointer semantics, as well as handling
                                 // destruction automatically.
  char init_buff_on_stack[DEFAULT_BUFFER_SIZE];
  printf_core::WriteBuffer wb(init_buff_on_stack, DEFAULT_BUFFER_SIZE,
                              resize_overflow_hook);
  printf_core::Writer writer(&wb);

  auto ret_val = printf_core::printf_main(&writer, format, args);
  if (ret_val < 0) {
    *ret = nullptr;
    return -1;
  }
  if (wb.buff == init_buff_on_stack) {
    *ret = static_cast<char *>(malloc(ret_val + 1));
    if (ret == nullptr) {
      return -1;
    }
    inline_memcpy(*ret, wb.buff, ret_val);
  } else {
    *ret = wb.buff;
  }
  (*ret)[ret_val] = '\0';
  return ret_val;
}

} // namespace LIBC_NAMESPACE
