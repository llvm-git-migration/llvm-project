//===--- CERTTidyModule.cpp - clang-tidy ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "../bugprone/BadSignalToKillThreadCheck.h"
#include "../bugprone/PointerArithmeticOnPolymorphicObjectCheck.h"
#include "../bugprone/ReservedIdentifierCheck.h"
#include "../bugprone/SignalHandlerCheck.h"
#include "../bugprone/SignedCharMisuseCheck.h"
#include "../bugprone/SizeofExpressionCheck.h"
#include "../bugprone/SpuriouslyWakeUpFunctionsCheck.h"
#include "../bugprone/SuspiciousMemoryComparisonCheck.h"
#include "../bugprone/UnhandledSelfAssignmentCheck.h"
#include "../bugprone/UnsafeFunctionsCheck.h"
#include "../bugprone/UnusedReturnValueCheck.h"
#include "../concurrency/ThreadCanceltypeAsynchronousCheck.h"
#include "../google/UnnamedNamespaceInHeaderCheck.h"
#include "../misc/NewDeleteOverloadsCheck.h"
#include "../misc/NonCopyableObjects.h"
#include "../misc/StaticAssertCheck.h"
#include "../misc/ThrowByValueCatchByReferenceCheck.h"
#include "../performance/MoveConstructorInitCheck.h"
#include "../readability/EnumInitialValueCheck.h"
#include "../readability/UppercaseLiteralSuffixCheck.h"
#include "CommandProcessorCheck.h"
#include "DefaultOperatorNewAlignmentCheck.h"
#include "DontModifyStdNamespaceCheck.h"
#include "FloatLoopCounter.h"
#include "LimitedRandomnessCheck.h"
#include "MutatingCopyCheck.h"
#include "NonTrivialTypesLibcMemoryCallsCheck.h"
#include "ProperlySeededRandomGeneratorCheck.h"
#include "SetLongJmpCheck.h"
#include "StaticObjectExceptionCheck.h"
#include "StrToNumCheck.h"
#include "ThrownExceptionTypeCheck.h"
#include "VariadicFunctionDefCheck.h"

namespace {

// Checked functions for cert-err33-c.
// The following functions are deliberately excluded because they can be called
// with NULL argument and in this case the check is not applicable:
// `mblen, mbrlen, mbrtowc, mbtowc, wctomb, wctomb_s`.
// FIXME: The check can be improved to handle such cases.
const llvm::StringRef CertErr33CCheckedFunctions = "^::aligned_alloc;"
                                                   "^::asctime_s;"
                                                   "^::at_quick_exit;"
                                                   "^::atexit;"
                                                   "^::bsearch;"
                                                   "^::bsearch_s;"
                                                   "^::btowc;"
                                                   "^::c16rtomb;"
                                                   "^::c32rtomb;"
                                                   "^::calloc;"
                                                   "^::clock;"
                                                   "^::cnd_broadcast;"
                                                   "^::cnd_init;"
                                                   "^::cnd_signal;"
                                                   "^::cnd_timedwait;"
                                                   "^::cnd_wait;"
                                                   "^::ctime_s;"
                                                   "^::fclose;"
                                                   "^::fflush;"
                                                   "^::fgetc;"
                                                   "^::fgetpos;"
                                                   "^::fgets;"
                                                   "^::fgetwc;"
                                                   "^::fopen;"
                                                   "^::fopen_s;"
                                                   "^::fprintf;"
                                                   "^::fprintf_s;"
                                                   "^::fputc;"
                                                   "^::fputs;"
                                                   "^::fputwc;"
                                                   "^::fputws;"
                                                   "^::fread;"
                                                   "^::freopen;"
                                                   "^::freopen_s;"
                                                   "^::fscanf;"
                                                   "^::fscanf_s;"
                                                   "^::fseek;"
                                                   "^::fsetpos;"
                                                   "^::ftell;"
                                                   "^::fwprintf;"
                                                   "^::fwprintf_s;"
                                                   "^::fwrite;"
                                                   "^::fwscanf;"
                                                   "^::fwscanf_s;"
                                                   "^::getc;"
                                                   "^::getchar;"
                                                   "^::getenv;"
                                                   "^::getenv_s;"
                                                   "^::gets_s;"
                                                   "^::getwc;"
                                                   "^::getwchar;"
                                                   "^::gmtime;"
                                                   "^::gmtime_s;"
                                                   "^::localtime;"
                                                   "^::localtime_s;"
                                                   "^::malloc;"
                                                   "^::mbrtoc16;"
                                                   "^::mbrtoc32;"
                                                   "^::mbsrtowcs;"
                                                   "^::mbsrtowcs_s;"
                                                   "^::mbstowcs;"
                                                   "^::mbstowcs_s;"
                                                   "^::memchr;"
                                                   "^::mktime;"
                                                   "^::mtx_init;"
                                                   "^::mtx_lock;"
                                                   "^::mtx_timedlock;"
                                                   "^::mtx_trylock;"
                                                   "^::mtx_unlock;"
                                                   "^::printf_s;"
                                                   "^::putc;"
                                                   "^::putwc;"
                                                   "^::raise;"
                                                   "^::realloc;"
                                                   "^::remove;"
                                                   "^::rename;"
                                                   "^::scanf;"
                                                   "^::scanf_s;"
                                                   "^::setlocale;"
                                                   "^::setvbuf;"
                                                   "^::signal;"
                                                   "^::snprintf;"
                                                   "^::snprintf_s;"
                                                   "^::sprintf;"
                                                   "^::sprintf_s;"
                                                   "^::sscanf;"
                                                   "^::sscanf_s;"
                                                   "^::strchr;"
                                                   "^::strerror_s;"
                                                   "^::strftime;"
                                                   "^::strpbrk;"
                                                   "^::strrchr;"
                                                   "^::strstr;"
                                                   "^::strtod;"
                                                   "^::strtof;"
                                                   "^::strtoimax;"
                                                   "^::strtok;"
                                                   "^::strtok_s;"
                                                   "^::strtol;"
                                                   "^::strtold;"
                                                   "^::strtoll;"
                                                   "^::strtoul;"
                                                   "^::strtoull;"
                                                   "^::strtoumax;"
                                                   "^::strxfrm;"
                                                   "^::swprintf;"
                                                   "^::swprintf_s;"
                                                   "^::swscanf;"
                                                   "^::swscanf_s;"
                                                   "^::thrd_create;"
                                                   "^::thrd_detach;"
                                                   "^::thrd_join;"
                                                   "^::thrd_sleep;"
                                                   "^::time;"
                                                   "^::timespec_get;"
                                                   "^::tmpfile;"
                                                   "^::tmpfile_s;"
                                                   "^::tmpnam;"
                                                   "^::tmpnam_s;"
                                                   "^::tss_create;"
                                                   "^::tss_get;"
                                                   "^::tss_set;"
                                                   "^::ungetc;"
                                                   "^::ungetwc;"
                                                   "^::vfprintf;"
                                                   "^::vfprintf_s;"
                                                   "^::vfscanf;"
                                                   "^::vfscanf_s;"
                                                   "^::vfwprintf;"
                                                   "^::vfwprintf_s;"
                                                   "^::vfwscanf;"
                                                   "^::vfwscanf_s;"
                                                   "^::vprintf_s;"
                                                   "^::vscanf;"
                                                   "^::vscanf_s;"
                                                   "^::vsnprintf;"
                                                   "^::vsnprintf_s;"
                                                   "^::vsprintf;"
                                                   "^::vsprintf_s;"
                                                   "^::vsscanf;"
                                                   "^::vsscanf_s;"
                                                   "^::vswprintf;"
                                                   "^::vswprintf_s;"
                                                   "^::vswscanf;"
                                                   "^::vswscanf_s;"
                                                   "^::vwprintf_s;"
                                                   "^::vwscanf;"
                                                   "^::vwscanf_s;"
                                                   "^::wcrtomb;"
                                                   "^::wcschr;"
                                                   "^::wcsftime;"
                                                   "^::wcspbrk;"
                                                   "^::wcsrchr;"
                                                   "^::wcsrtombs;"
                                                   "^::wcsrtombs_s;"
                                                   "^::wcsstr;"
                                                   "^::wcstod;"
                                                   "^::wcstof;"
                                                   "^::wcstoimax;"
                                                   "^::wcstok;"
                                                   "^::wcstok_s;"
                                                   "^::wcstol;"
                                                   "^::wcstold;"
                                                   "^::wcstoll;"
                                                   "^::wcstombs;"
                                                   "^::wcstombs_s;"
                                                   "^::wcstoul;"
                                                   "^::wcstoull;"
                                                   "^::wcstoumax;"
                                                   "^::wcsxfrm;"
                                                   "^::wctob;"
                                                   "^::wctrans;"
                                                   "^::wctype;"
                                                   "^::wmemchr;"
                                                   "^::wprintf_s;"
                                                   "^::wscanf;"
                                                   "^::wscanf_s;";

} // namespace

namespace clang::tidy {
namespace cert {

class CERTModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    // C++ checkers
    // CON
    CheckFactories.registerCheck<bugprone::SpuriouslyWakeUpFunctionsCheck>(
        "cert-con54-cpp");
    // CTR
    CheckFactories
        .registerCheck<bugprone::PointerArithmeticOnPolymorphicObjectCheck>(
            "cert-ctr56-cpp");
    // DCL
    CheckFactories.registerCheck<VariadicFunctionDefCheck>("cert-dcl50-cpp");
    CheckFactories.registerCheck<bugprone::ReservedIdentifierCheck>(
        "cert-dcl51-cpp");
    CheckFactories.registerCheck<misc::NewDeleteOverloadsCheck>(
        "cert-dcl54-cpp");
    CheckFactories.registerCheck<DontModifyStdNamespaceCheck>(
        "cert-dcl58-cpp");
    CheckFactories.registerCheck<google::build::UnnamedNamespaceInHeaderCheck>(
        "cert-dcl59-cpp");
    // ERR
    CheckFactories.registerCheck<misc::ThrowByValueCatchByReferenceCheck>(
        "cert-err09-cpp");
    CheckFactories.registerCheck<SetLongJmpCheck>("cert-err52-cpp");
    CheckFactories.registerCheck<StaticObjectExceptionCheck>("cert-err58-cpp");
    CheckFactories.registerCheck<ThrownExceptionTypeCheck>("cert-err60-cpp");
    CheckFactories.registerCheck<misc::ThrowByValueCatchByReferenceCheck>(
        "cert-err61-cpp");
    // MEM
    CheckFactories.registerCheck<DefaultOperatorNewAlignmentCheck>(
        "cert-mem57-cpp");
    // MSC
    CheckFactories.registerCheck<LimitedRandomnessCheck>("cert-msc50-cpp");
    CheckFactories.registerCheck<ProperlySeededRandomGeneratorCheck>(
        "cert-msc51-cpp");
    CheckFactories.registerCheck<bugprone::SignalHandlerCheck>(
        "cert-msc54-cpp");
    // OOP
    CheckFactories.registerCheck<performance::MoveConstructorInitCheck>(
        "cert-oop11-cpp");
    CheckFactories.registerCheck<bugprone::UnhandledSelfAssignmentCheck>(
        "cert-oop54-cpp");
    CheckFactories.registerCheck<NonTrivialTypesLibcMemoryCallsCheck>(
        "cert-oop57-cpp");
    CheckFactories.registerCheck<MutatingCopyCheck>(
        "cert-oop58-cpp");

    // C checkers
    // ARR
    CheckFactories.registerCheck<bugprone::SizeofExpressionCheck>(
        "cert-arr39-c");
    // CON
    CheckFactories.registerCheck<bugprone::SpuriouslyWakeUpFunctionsCheck>(
        "cert-con36-c");
    // DCL
    CheckFactories.registerCheck<misc::StaticAssertCheck>("cert-dcl03-c");
    CheckFactories.registerCheck<readability::UppercaseLiteralSuffixCheck>(
        "cert-dcl16-c");
    CheckFactories.registerCheck<bugprone::ReservedIdentifierCheck>(
        "cert-dcl37-c");
    // ENV
    CheckFactories.registerCheck<CommandProcessorCheck>("cert-env33-c");
    // ERR
    CheckFactories.registerCheck<bugprone::UnusedReturnValueCheck>(
        "cert-err33-c");
    CheckFactories.registerCheck<StrToNumCheck>("cert-err34-c");
    // EXP
    CheckFactories.registerCheck<bugprone::SuspiciousMemoryComparisonCheck>(
        "cert-exp42-c");
    // FLP
    CheckFactories.registerCheck<FloatLoopCounter>("cert-flp30-c");
    CheckFactories.registerCheck<bugprone::SuspiciousMemoryComparisonCheck>(
        "cert-flp37-c");
    // FIO
    CheckFactories.registerCheck<misc::NonCopyableObjectsCheck>("cert-fio38-c");
    // INT
    CheckFactories.registerCheck<readability::EnumInitialValueCheck>(
        "cert-int09-c");
    // MSC
    CheckFactories.registerCheck<bugprone::UnsafeFunctionsCheck>(
        "cert-msc24-c");
    CheckFactories.registerCheck<LimitedRandomnessCheck>("cert-msc30-c");
    CheckFactories.registerCheck<ProperlySeededRandomGeneratorCheck>(
        "cert-msc32-c");
    CheckFactories.registerCheck<bugprone::UnsafeFunctionsCheck>(
        "cert-msc33-c");
    // POS
    CheckFactories.registerCheck<bugprone::BadSignalToKillThreadCheck>(
        "cert-pos44-c");
    CheckFactories
        .registerCheck<concurrency::ThreadCanceltypeAsynchronousCheck>(
            "cert-pos47-c");
    // SIG
    CheckFactories.registerCheck<bugprone::SignalHandlerCheck>("cert-sig30-c");
    // STR
    CheckFactories.registerCheck<bugprone::SignedCharMisuseCheck>(
        "cert-str34-c");
    // temp

    CheckFactories.registerCheck<bugprone::UnsafeFunctionsCheck>(
        "ericsson-unsafe-functions");
  }

  ClangTidyOptions getModuleOptions() override {
    ClangTidyOptions Options;
    ClangTidyOptions::OptionMap &Opts = Options.CheckOptions;
    Opts["cert-arr39-c.WarnOnSizeOfConstant"] = "false";
    Opts["cert-arr39-c.WarnOnSizeOfIntegerExpression"] = "false";
    Opts["cert-arr39-c.WarnOnSizeOfThis"] = "false";
    Opts["cert-arr39-c.WarnOnSizeOfCompareToConstant"] = "false";
    Opts["cert-arr39-c.WarnOnSizeOfPointer"] = "false";
    Opts["cert-arr39-c.WarnOnSizeOfPointerToAggregate"] = "false";
    Opts["cert-dcl16-c.NewSuffixes"] = "L;LL;LU;LLU";
    Opts["cert-err33-c.CheckedFunctions"] = CertErr33CCheckedFunctions;
    Opts["cert-err33-c.AllowCastToVoid"] = "true";
    Opts["cert-oop54-cpp.WarnOnlyIfThisHasSuspiciousField"] = "false";
    Opts["cert-str34-c.DiagnoseSignedUnsignedCharComparisons"] = "false";
    Opts["ericsson-unsafe-functions.ReportDefaultFunctions"] = "false";
    Opts["ericsson-unsafe-functions.CustomFunctions"] =
        // High priority
        "^::gets$,fgets,is insecure and deprecated;"
        "^::sprintf$,snprintf,is insecure and deprecated;"
        "^::strcpy$,strlcpy,is insecure and deprecated;"
        "^::strncpy$,strlcpy,is insecure and deprecated;"
        "^::vsprintf,vsnprintf,is insecure and deprecated;"
        "^::wcscat$,wcslcat,is insecure and deprecated;"
        "^::wcscpy$,wcslcpy,is insecure and deprecated;"
        "^::wcsncat$,wcslcat,is insecure and deprecated;"
        "^::wcsncpy$,wcslcpy,is insecure and deprecated;"

        "^::bcopy$,memcpy or memmove,is insecure and deprecated;"
        "^::bzero$,memset,is insecure and deprecated;"
        "^::index$,strchr,is insecure and deprecated;"
        "^::rindex$,strrchr,is insecure and deprecated;"

        "^::valloc$,aligned_alloc,is insecure and deprecated;"

        "^::tmpnam$,mkstemp,is insecure and deprecated;"
        "^::tmpnam_r$,mkstemp,is insecure and deprecated;"

        "^::getwd$,getcwd,is insecure and deprecated;"
        "^::crypt$,an Ericsson-recommended crypto algorithm,is insecure and "
        "deprecated;"
        "^::encrypt$,an Ericsson-recommended crypto algorithm,is insecure and "
        "deprecated;"
        "^::stpcpy$,strlcpy,is insecure and deprecated;"
        "^::strcat$,strlcat,is insecure and deprecated;"
        "^::strncat$,strlcat,is insecure and deprecated;"

        // Medium priority
        "^::scanf$,strto__,lacks error detection;"
        "^::fscanf$,strto__,lacks error detection;"
        "^::sscanf$,strto__,lacks error detection;"
        "^::vscanf$,strto__,lacks error detection;"
        "^::vsscanf$,strto__,lacks error detection;"

        "^::atof$,strtod,lacks error detection;"
        "^::atof_l$,strtod,lacks error detection;"
        "^::atoi$,strtol,lacks error detection;"
        "^::atoi_l$,strtol,lacks error detection;"
        "^::atol$,strtol,lacks error detection;"
        "^::atol_l$,strtol,lacks error detection;"
        "^::atoll$,strtoll,lacks error detection;"
        "^::atoll_l$,strtoll,lacks error detection;"
        "^::setbuf$,setvbuf,lacks error detection;"
        "^::setjmp$,,lacks error detection;"
        "^::sigsetjmp$,,lacks error detection;"
        "^::longjmp$,,lacks error detection;"
        "^::siglongjmp$,,lacks error detection;"
        "^::rewind$,fseek,lacks error detection;"

        // Low priority
        "^::strtok$,strtok_r,is not thread safe;"
        "^::strerror$,strerror_r,is not thread safe;"
        "^::wcstombs$,_FORTIFY_SOURCE,is recommended to have source hardening;"
        "^::mbstowcs$,_FORTIFY_SOURCE,is recommended to have source hardening;"

        "^::getenv$,,is not thread safe;"
        "^::mktemp$,mkstemp,is not thread safe;"
        "^::perror$,,is not thread safe;"

        "^::access$,,is not thread safe;"
        "^::asctime$,asctime_r,is not thread safe;"
        "^::atomic_init$,,is not thread safe;"
        "^::c16rtomb$,,is not thread safe;"
        "^::c32rtomb$,,is not thread safe;"
        "^::catgets$,,is not thread safe;"
        "^::ctermid$,,is not thread safe;"
        "^::ctime$,cmtime_r,is not thread safe;"
        "^::dbm_clearerr$,a database client library,is not thread safe;"
        "^::dbm_close$,a database client library,is not thread safe;"
        "^::dbm_delete$,a database client library,is not thread safe;"
        "^::dbm_error$,a database client library,is not thread safe;"
        "^::dbm_fetch$,a database client library,is not thread safe;"
        "^::dbm_firstkey$,a database client library,is not thread safe;"
        "^::dbm_nextkey$,a database client library,is not thread safe;"
        "^::dbm_open$,a database client library,is not thread safe;"
        "^::dbm_store$,a database client library,is not thread safe;"
        "^::dlerror$,,is not thread safe;"
        "^::drand48$,drand48_r,is not thread safe;"
        "^::endgrent$,,is not thread safe;"
        "^::endpwent$,,is not thread safe;"
        "^::endutxent$,,is not thread safe;"
        "^::getc_unlocked$,,is not thread safe;"
        "^::getchar_unlocked$,,is not thread safe;"
        "^::getdate$,,is not thread safe;"
        "^::getgrent$,,is not thread safe;"
        "^::getgrgid$,,is not thread safe;"
        "^::getgrnam$,,is not thread safe;"
        "^::gethostent$,,is not thread safe;"
        "^::getlogin$,,is not thread safe;"
        "^::getnetbyaddr$,,is not thread safe;"
        "^::getnetbyname$,,is not thread safe;"
        "^::getnetent$,,is not thread safe;"
        "^::getopt$,,is not thread safe;"
        "^::getprotobyname$,,is not thread safe;"
        "^::getprotobynumber$,,is not thread safe;"
        "^::getprotoent$,,is not thread safe;"
        "^::getpwent$,,is not thread safe;"
        "^::getpwnam$,,is not thread safe;"
        "^::getpwuid$,,is not thread safe;"
        "^::getservbyname$,,is not thread safe;"
        "^::getservbyport$,,is not thread safe;"
        "^::getservent$,,is not thread safe;"
        "^::getutxent$,,is not thread safe;"
        "^::getutxid$,,is not thread safe;"
        "^::getutxline$,,is not thread safe;"
        "^::gmtime$,gmtime_r,is not thread safe;"
        "^::hcreate$,,is not thread safe;"
        "^::hdestroy$,,is not thread safe;"
        "^::hsearch$,,is not thread safe;"
        "^::inet_ntoa$,,is not thread safe;"
        "^::l64a$,,is not thread safe;"
        "^::lgamma$,,is not thread safe;"
        "^::lgammaf$,,is not thread safe;"
        "^::lgammal$,,is not thread safe;"
        "^::localeconv$,,is not thread safe;"
        "^::localtime$,localtime_r,is not thread safe;"
        "^::lrand48$,lrand48_r,is not thread safe;"
        "^::mblen$,,is not thread safe;"
        "^::mbrto16$,,is not thread safe;"
        "^::mbrto32$,,is not thread safe;"
        "^::mbrtowc$,,is not thread safe;"
        "^::mbsnrtowcs$,,is not thread safe;"
        "^::mbsrtowcs$,,is not thread safe;"
        "^::mrand48$,mrand48_r,is not thread safe;"
        "^::nftw$,,is not thread safe;"
        "^::ni_langinfo$,,is not thread safe;"
        "^::ptsname$,,is not thread safe;"
        "^::putc_unlocked$,,is not thread safe;"
        "^::putchar_unlocked$,,is not thread safe;"
        "^::putenv$,,is not thread safe;"
        "^::pututxline$,,is not thread safe;"
        "^::rand$,,is not thread safe;"
        "^::readdir$,,is not thread safe;"
        "^::setenv$,,is not thread safe;"
        "^::setgrent$,,is not thread safe;"
        "^::setkey$,an Ericsson-recommended crypto algorithm,is insecure, "
        "deprecated, and not thread safe;"
        "^::setlocale$,,is not thread safe;"
        "^::setpwent$,,is not thread safe;"
        "^::setutxent$,,is not thread safe;"
        "^::srand$,,is not thread safe;"
        "^::strsignal$,,is not thread safe;"
        "^::ttyname$,,is not thread safe;"
        "^::unsetenv$,,is not thread safe;"
        "^::wctomb$,,is not thread safe;"
        "^::wcrtomb$,,is not thread safe;"
        "^::wcsnrtombs$,,is not thread safe;"
        "^::wcsrtombs$,,is not thread safe;";
    return Options;
  }
};

} // namespace cert

// Register the MiscTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<cert::CERTModule>
    X("cert-module",
      "Adds lint checks corresponding to CERT secure coding guidelines.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the CERTModule.
volatile int CERTModuleAnchorSource = 0;

} // namespace clang::tidy
