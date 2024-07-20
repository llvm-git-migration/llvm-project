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
  char *TmpBuf =
      static_cast<char *>(realloc(wb->buff, new_size + 1)); // +1 for null
  if (TmpBuf == nullptr) {
    return -1;
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
  auto init_buff = static_cast<char *>(malloc(DEFAULT_BUFFER_SIZE));
  if (init_buff == nullptr)
    return -1;
  printf_core::WriteBuffer wb(init_buff, DEFAULT_BUFFER_SIZE,
                              resize_overflow_hook);
  printf_core::Writer writer(&wb);
  int ret_val = printf_core::printf_main(&writer, format, args);

  if (ret_val == printf_core::FILE_WRITE_ERROR) {
    // init_buff should have been freed during successful realloc.
    if (wb.buff != init_buff) {
      free(wb.buff);
    } else {
      free(init_buff);
    }
    *ret = nullptr;
    return -1;
  }

  const bool need_shrink = (wb.buff == init_buff) &&
                           (wb.buff_len == static_cast<size_t>(ret_val) + 1);
  *ret = (need_shrink) ? (static_cast<char *>(realloc(init_buff, ret_val + 1)))
                       : wb.buff;
  if (LIBC_UNLIKELY(ret == nullptr)) {
    free(init_buff);
    return -1;
  }

  (*ret)[ret_val] = '\0';
  return ret_val;
}

} // namespace LIBC_NAMESPACE
