#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_DANGLING_REFERENCE_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_DANGLING_REFERENCE_H
#include "clang/AST/DeclBase.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Sema/Sema.h"

namespace clang {
void runDanglingReferenceAnalysis(const DeclContext &dc, const CFG &cfg,
                                  AnalysisDeclContext &ac, Sema &S);

} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_DANGLING_REFERENCE_H
