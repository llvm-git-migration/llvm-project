//===-- Implementation header for strfromx() utilitites -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// According to the C23 standard, any input character sequences except a
// precision specifier and the usual floating point formats, namely
// %{a,A,e,E,f,F,g,G}, are not allowed and any code that does otherwise results
// in undefined behaviour; which in this case is that the buffer string is
// simply populated with the format string. The case of the value of buffer size
// being 0 or the input being NULL should be handled in the calling function
// (strfromf, strfromd, strfroml) itself.

#ifndef LLVM_LIBC_SRC_STDLIB_STRFROM_UTIL_H
#define LLVM_LIBC_SRC_STDLIB_STRFROM_UTIL_H

#include "src/__support/str_to_integer.h"
#include "src/stdio/printf_core/converter_atlas.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/writer.h"

#include <stddef.h>

namespace LIBC_NAMESPACE::internal {

template <typename T> struct type_of {
  using type = T;
};

template <> struct type_of<float> {
  using type = fputil::FPBits<float>::StorageType;
};
template <> struct type_of<double> {
  using type = fputil::FPBits<double>::StorageType;
};
template <> struct type_of<long double> {
  using type = fputil::FPBits<long double>::StorageType;
};

template <typename T> using type_of_v = typename type_of<T>::type;

template <typename T>
printf_core::FormatSection parse_format_string(const char *__restrict format,
                                               T fp) {
  printf_core::FormatSection section;
  size_t cur_pos = 0;

  if (format[cur_pos] == '%') {
    section.has_conv = true;
    ++cur_pos;

    // handle precision
    section.precision = -1;
    if (format[cur_pos] == '.') {
      ++cur_pos;
      section.precision = 0;

      // The standard does not allow the '*' (asterisk) operator for strfromx()
      // functions
      if (internal::isdigit(format[cur_pos])) {
        auto result = internal::strtointeger<int>(format + cur_pos, 10);
        section.precision += result.value;
        cur_pos += result.parsed_len;
      }
    }

    section.conv_name = format[cur_pos];
    switch (format[cur_pos]) {
    case '%':
      section.has_conv = true;
      break;
    case 'a':
    case 'A':
    case 'e':
    case 'E':
    case 'f':
    case 'F':
    case 'g':
    case 'G':
      section.conv_val_raw = cpp::bit_cast<type_of_v<T>>(fp);
      break;
    default:
      // error out, invalid format specifier
      section.has_conv = false;
      while (format[cur_pos] != '\0')
        ++cur_pos;
      break;
    }

    if (format[cur_pos] != '\0')
      ++cur_pos;
  } else {
    section.has_conv = false;
    // We are looking for exactly one section, so no more '%'
    while (format[cur_pos] != '\0')
      ++cur_pos;
  }

  section.raw_string = {format, cur_pos};
  return section;
}

int convert(const printf_core::FormatSection &section,
            printf_core::Writer *writer) {
  if (!section.has_conv)
    return writer->write(section.raw_string);

  switch (section.conv_name) {
  case '%':
    return writer->write("%");
  case 'f':
  case 'F':
    return convert_float_decimal(writer, section);
  case 'e':
  case 'E':
    return convert_float_dec_exp(writer, section);
  case 'a':
  case 'A':
    return convert_float_hex_exp(writer, section);
  case 'g':
  case 'G':
    return convert_float_dec_auto(writer, section);
  default:
    return writer->write(section.raw_string);
  }
  return -1;
}

} // namespace LIBC_NAMESPACE::internal

#endif // LLVM_LIBC_SRC_STDLIB_STRFROM_UTIL_H
