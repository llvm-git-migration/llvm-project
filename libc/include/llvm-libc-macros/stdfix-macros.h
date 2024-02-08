//===-- Definitions from stdfix.h -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_STDFIX_MACROS_H
#define __LLVM_LIBC_MACROS_STDFIX_MACROS_H

#ifdef __clang__
#if (!defined(__cplusplus) || (__clang_major__ >= 18))
// _Fract and _Accum types are avaiable
#define LIBC_COMPILER_HAS_FIXED_POINT
#endif // __cplusplus
#endif // __clang__

#ifdef LIBC_COMPILER_HAS_FIXED_POINT

#ifdef __SFRACT_FBIT__
#define SFRACT_FBIT __SFRACT_FBIT__
#else
#define SFRACT_FBIT 7
#endif // SFRACT_FBIT

#ifdef __SFRACT_MIN__
#define SFRACT_MIN __SFRACT_MIN__
#else
#define SFRACT_MIN (-0.5HR - 0.5HR)
#endif // SFRACT_MIN

#ifdef __SFRACT_MAX__
#define SFRACT_MAX __SFRACT_MAX__
#else
#define SFRACT_MAX 0x1.FCp-1HR
#endif // SFRACT_MAX

#ifdef __SFRACT_EPSILON__
#define SFRACT_EPSILON __SFRACT_EPSILON__
#else
#define SFRACT_EPSILON 0x1p-7HR
#endif // SFRACT_EPSILON

#ifdef __USFRACT_FBIT__
#define USFRACT_FBIT __USFRACT_FBIT__
#else
#define USFRACT_FBIT 7
#endif // USFRACT_FBIT

#ifdef __USFRACT_MIN__
#define USFRACT_MIN __USFRACT_MIN__
#else
#define USFRACT_MIN 0UHR
#endif // USFRACT_MIN

#ifdef __USFRACT_MAX__
#define USFRACT_MAX __USFRACT_MAX__
#else
#define USFRACT_MAX 0x1.FCp-1UHR
#endif // USFRACT_MAX

#ifdef __USFRACT_EPSILON__
#define USFRACT_EPSILON __USFRACT_EPSILON__
#else
#define USFRACT_EPSILON 0x1p-7UHR
#endif // USFRACT_EPSILON

#ifdef __FRACT_FBIT__
#define FRACT_FBIT __FRACT_FBIT__
#else
#define FRACT_FBIT 15
#endif // FRACT_FBIT

#ifdef __FRACT_MIN__
#define FRACT_MIN __FRACT_MIN__
#else
#define FRACT_MIN (-0.5R - 0.5R)
#endif // FRACT_MIN

#ifdef __FRACT_MAX__
#define FRACT_MAX __FRACT_MAX__
#else
#define FRACT_MAX 0x1.FFFCp-1R
#endif // FRACT_MAX

#ifdef __FRACT_EPSILON__
#define FRACT_EPSILON __FRACT_EPSILON__
#else
#define FRACT_EPSILON 0x1p-15R
#endif // FRACT_EPSILON

#ifdef __UFRACT_FBIT__
#define UFRACT_FBIT __UFRACT_FBIT__
#else
#define UFRACT_FBIT 15
#endif // UFRACT_FBIT

#ifdef __UFRACT_MIN__
#define UFRACT_MIN __UFRACT_MIN__
#else
#define UFRACT_MIN 0UR
#endif // UFRACT_MIN

#ifdef __UFRACT_MAX__
#define UFRACT_MAX __UFRACT_MAX__
#else
#define UFRACT_MAX 0x1.FFFCp-1UR
#endif // UFRACT_MAX

#ifdef __UFRACT_EPSILON__
#define UFRACT_EPSILON __UFRACT_EPSILON__
#else
#define UFRACT_EPSILON 0x1p-15UR
#endif // UFRACT_EPSILON

#ifdef __LFRACT_FBIT__
#define LFRACT_FBIT __LFRACT_FBIT__
#else
#define LFRACT_FBIT 23
#endif // LFRACT_FBIT

#ifdef __LFRACT_MIN__
#define LFRACT_MIN __LFRACT_MIN__
#else
#define LFRACT_MIN (-0.5LR - 0.5LR)
#endif // LFRACT_MIN

#ifdef __LFRACT_MAX__
#define LFRACT_MAX __LFRACT_MAX__
#else
#define LFRACT_MAX 0x1.FFFFFCp-1LR
#endif // LFRACT_MAX

#ifdef __LFRACT_EPSILON__
#define LFRACT_EPSILON __LFRACT_EPSILON__
#else
#define LFRACT_EPSILON 0x1p-23LR
#endif // LFRACT_EPSILON

#ifdef __ULFRACT_FBIT__
#define ULFRACT_FBIT __ULFRACT_FBIT__
#else
#define ULFRACT_FBIT 23
#endif // ULFRACT_FBIT

#ifdef __ULFRACT_MIN__
#define ULFRACT_MIN __ULFRACT_MIN__
#else
#define ULFRACT_MIN 0ULR
#endif // ULFRACT_MIN

#ifdef __ULFRACT_MAX__
#define ULFRACT_MAX __ULFRACT_MAX__
#else
#define ULFRACT_MAX 0x1.FFFFFCp-1ULR
#endif // ULFRACT_MAX

#ifdef __ULFRACT_EPSILON__
#define ULFRACT_EPSILON __ULFRACT_EPSILON__
#else
#define ULFRACT_EPSILON 0x1p-23ULR
#endif // ULFRACT_EPSILON

#ifdef __SACCUM_FBIT__
#define SACCUM_FBIT __SACCUM_FBIT__
#else
#define SACCUM_FBIT 7
#endif // SACCUM_FBIT

#ifdef __SACCUM_IBIT__
#define SACCUM_IBIT __SACCUM_IBIT__
#else
#define SACCUM_IBIT 4
#endif // SACCUM_IBIT

#ifdef __SACCUM_MIN__
#define SACCUM_MIN __SACCUM_MIN__
#else
#define SACCUM_MIN (-8HK - 8HK)
#endif // SACCUM_MIN

#ifdef __SACCUM_MAX__
#define SACCUM_MAX __SACCUM_MAX__
#else
#define SACCUM_MAX 0x1.FFCp+3HK
#endif // SACCUM_MAX

#ifdef __SACCUM_EPSILON__
#define SACCUM_EPSILON __SACCUM_EPSILON__
#else
#define SACCUM_EPSILON 0x1p-7HK
#endif // SACCUM_EPSILON

#ifdef __USACCUM_FBIT__
#define USACCUM_FBIT __USACCUM_FBIT__
#else
#define USACCUM_FBIT 7
#endif // USACCUM_FBIT

#ifdef __USACCUM_IBIT__
#define USACCUM_IBIT __USACCUM_IBIT__
#else
#define USACCUM_IBIT 4
#endif // USACCUM_IBIT

#ifdef __USACCUM_MIN__
#define USACCUM_MIN __USACCUM_MIN__
#else
#define USACCUM_MIN 0UHK
#endif // USACCUM_MIN

#ifdef __USACCUM_MAX__
#define USACCUM_MAX __USACCUM_MAX__
#else
#define USACCUM_MAX 0x1.FFCp+3UHK
#endif // USACCUM_MAX

#ifdef __USACCUM_EPSILON__
#define USACCUM_EPSILON __USACCUM_EPSILON__
#else
#define USACCUM_EPSILON 0x1p-7UHK
#endif // USACCUM_EPSILON

#ifdef __ACCUM_FBIT__
#define ACCUM_FBIT __ACCUM_FBIT__
#else
#define ACCUM_FBIT 15
#endif // ACCUM_FBIT

#ifdef __ACCUM_IBIT__
#define ACCUM_IBIT __ACCUM_IBIT__
#else
#define ACCUM_IBIT 4
#endif // ACCUM_IBIT

#ifdef __ACCUM_MIN__
#define ACCUM_MIN __ACCUM_MIN__
#else
#define ACCUM_MIN (-8R - 8R)
#endif // ACCUM_MIN

#ifdef __ACCUM_MAX__
#define ACCUM_MAX __ACCUM_MAX__
#else
#define ACCUM_MAX 0x1.FFFFCp+3K
#endif // ACCUM_MAX

#ifdef __ACCUM_EPSILON__
#define ACCUM_EPSILON __ACCUM_EPSILON__
#else
#define ACCUM_EPSILON 0x1p-15K
#endif // ACCUM_EPSILON

#ifdef __UACCUM_FBIT__
#define UACCUM_FBIT __UACCUM_FBIT__
#else
#define UACCUM_FBIT 15
#endif // UACCUM_FBIT

#ifdef __UACCUM_IBIT__
#define UACCUM_IBIT __UACCUM_IBIT__
#else
#define UACCUM_IBIT 4
#endif // UACCUM_IBIT

#ifdef __UACCUM_MIN__
#define UACCUM_MIN __UACCUM_MIN__
#else
#define UACCUM_MIN 0UK
#endif // UACCUM_MIN

#ifdef __UACCUM_MAX__
#define UACCUM_MAX __UACCUM_MAX__
#else
#define UACCUM_MAX 0x1.FFFFCp+3UK
#endif // UACCUM_MAX

#ifdef __UACCUM_EPSILON__
#define UACCUM_EPSILON __UACCUM_EPSILON__
#else
#define UACCUM_EPSILON 0x1p-15UK
#endif // UACCUM_EPSILON

#ifdef __LACCUM_FBIT__
#define LACCUM_FBIT __LACCUM_FBIT__
#else
#define LACCUM_FBIT 23
#endif // LACCUM_FBIT

#ifdef __LACCUM_IBIT__
#define LACCUM_IBIT __LACCUM_IBIT__
#else
#define LACCUM_IBIT 4
#endif // LACCUM_IBIT

#ifdef __LACCUM_MIN__
#define LACCUM_MIN __LACCUM_MIN__
#else
#define LACCUM_MIN (-8LK - 8LK)
#endif // LACCUM_MIN

#ifdef __LACCUM_MAX__
#define LACCUM_MAX __LACCUM_MAX__
#else
#define LACCUM_MAX 0x1.FFFFFFCp+3LK
#endif // LACCUM_MAX

#ifdef __LACCUM_EPSILON__
#define LACCUM_EPSILON __LACCUM_EPSILON__
#else
#define LACCUM_EPSILON 0x1p-23LK
#endif // LACCUM_EPSILON

#ifdef __ULACCUM_FBIT__
#define ULACCUM_FBIT __ULACCUM_FBIT__
#else
#define ULACCUM_FBIT 23
#endif // ULACCUM_FBIT

#ifdef __ULACCUM_IBIT__
#define ULACCUM_IBIT __ULACCUM_IBIT__
#else
#define ULACCUM_IBIT 4
#endif // ULACCUM_IBIT

#ifdef __ULACCUM_MIN__
#define ULACCUM_MIN __ULACCUM_MIN__
#else
#define ULACCUM_MIN 0ULK
#endif // ULACCUM_MIN

#ifdef __ULACCUM_MAX__
#define ULACCUM_MAX __ULACCUM_MAX__
#else
#define ULACCUM_MAX 0x1.FFFFFFCp+3ULK
#endif // ULACCUM_MAX

#ifdef __ULACCUM_EPSILON__
#define ULACCUM_EPSILON __ULACCUM_EPSILON__
#else
#define ULACCUM_EPSILON 0x1p-23ULK
#endif // ULACCUM_EPSILON

#endif // LIBC_COMPILER_HAS_FIXED_POINT

#endif // __LLVM_LIBC_MACROS_STDFIX_MACROS_H
