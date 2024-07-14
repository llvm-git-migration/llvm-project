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

LIBC_INLINE int resize_overflow_hook(printf_core::WriteBuffer *wb,
                                     cpp::string_view new_str) {
  size_t NewSize = new_str.size() + wb->buff_cur;
  char *TmpBuf =
      static_cast<char *>(realloc(wb->buff, NewSize + 1)); // +1 for null
  if (TmpBuf == nullptr) {
    return printf_core::FILE_WRITE_ERROR;
  }
  wb->buff = TmpBuf;
  inline_memcpy(wb->buff + wb->buff_cur, new_str.data(), new_str.size());
  wb->buff_cur = NewSize;
  wb->buff_len = NewSize;
  return printf_core::WRITE_OK;
}

} // namespace

LLVM_LIBC_FUNCTION(int, vasprintf,
                   (char **__restrict ret, const char *format, va_list vlist)) {
  internal::ArgList args(vlist); // This holder class allows for easier copying
                                 // and pointer semantics, as well as handling
                                 // destruction automatically.
  const uint16_t defaultSize = 200;
  auto InitBuff = static_cast<char *>(malloc(defaultSize));
  if (InitBuff == nullptr)
    return printf_core::FILE_WRITE_ERROR;
  printf_core::WriteBuffer wb(InitBuff, defaultSize, resize_overflow_hook);
  printf_core::Writer writer(&wb);
  int ret_val = printf_core::printf_main(&writer, format, args);
  if (LIBC_LIKELY(wb.buff == InitBuff)) {
    *ret = static_cast<char *>(realloc(InitBuff, ret_val + 1)); // +1 for null
    if (LIBC_UNLIKELY(ret == nullptr)) {
      free(InitBuff);
      return printf_core::FILE_WRITE_ERROR;
    }
  }
  if (ret_val == printf_core::FILE_WRITE_ERROR) {
    if (wb.buff != InitBuff) {
      free(wb.buff); // InitBuff should have been freed during successful
                     // realloc.
    } else {
      free(InitBuff);
    }
    return printf_core::FILE_WRITE_ERROR;
  }
  (*ret)[ret_val] = '\0';
  return ret_val;
}

} // namespace LIBC_NAMESPACE
