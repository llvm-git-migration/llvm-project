//===- llvm/Analysis/ConditionCacheUtil.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared by DomConditionCache and AssumptionCache. Holds common operation of
// finding values potentially affected by an assumed/branched on condition.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CONDITIONCACHEUTIL_H
#define LLVM_ANALYSIS_CONDITIONCACHEUTIL_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/PatternMatch.h"
#include <functional>

namespace llvm {

static void addValueAffectedByCondition(
    Value *V, function_ref<void(Value *, int)> InsertAffected, int Idx = -1) {
  using namespace llvm::PatternMatch;
  assert(V != nullptr);
  if (isa<Argument>(V) || isa<GlobalValue>(V)) {
    InsertAffected(V, Idx);
  } else if (auto *I = dyn_cast<Instruction>(V)) {
    InsertAffected(V, Idx);

    // Peek through unary operators to find the source of the condition.
    Value *Op;
    if (match(I, m_PtrToInt(m_Value(Op)))) {
      if (isa<Instruction>(Op) || isa<Argument>(Op))
        InsertAffected(Op, Idx);
    }
  }
}

static void
findValuesAffectedByCondition(Value *Cond, bool IsAssume,
                              function_ref<void(Value *, int)> InsertAffected) {
  using namespace llvm::PatternMatch;
  auto AddAffected = [&InsertAffected](Value *V) {
    addValueAffectedByCondition(V, InsertAffected);
  };

  SmallVector<Value *, 8> Worklist;
  SmallPtrSet<Value *, 8> Visited;
  Worklist.push_back(Cond);
  while (!Worklist.empty()) {
    Value *V = Worklist.pop_back_val();
    if (!Visited.insert(V).second)
      continue;

    CmpInst::Predicate Pred;
    Value *A, *B, *X;

    if (IsAssume) {
      AddAffected(V);
      if (match(V, m_Not(m_Value(X))))
        AddAffected(X);
    }

    if (match(V, m_LogicalOp(m_Value(A), m_Value(B)))) {
      Worklist.push_back(A);
      Worklist.push_back(B);
    } else if (match(V, m_Cmp(Pred, m_Value(A), m_Value(B)))) {
      if (IsAssume) {
        AddAffected(A);
        AddAffected(B);
      } else if (match(B, m_Constant()))
        AddAffected(A);

      if (ICmpInst::isEquality(Pred)) {
        if (match(B, m_ConstantInt())) {
          // (X & C) or (X | C) or (X ^ C).
          // (X << C) or (X >>_s C) or (X >>_u C).
          if (match(A, m_BitwiseLogic(m_Value(X), m_ConstantInt())) ||
              match(A, m_Shift(m_Value(X), m_ConstantInt())))
            AddAffected(X);
        }
      } else {
        // Handle (A + C1) u< C2, which is the canonical form of
        // A > C3 && A < C4.
        if (match(A, m_Add(m_Value(X), m_ConstantInt())) &&
            match(B, m_ConstantInt()))
          AddAffected(X);

        // Handle icmp slt/sgt (bitcast X to int), 0/-1, which is supported
        // by computeKnownFPClass().
        if (match(A, m_ElementWiseBitCast(m_Value(X)))) {
          if (Pred == ICmpInst::ICMP_SLT && match(B, m_Zero()))
            InsertAffected(X, -1);
          else if (Pred == ICmpInst::ICMP_SGT && match(B, m_AllOnes()))
            InsertAffected(X, -1);
        }

        if (CmpInst::isFPPredicate(Pred)) {
          // fcmp fneg(x), y
          // fcmp fabs(x), y
          // fcmp fneg(fabs(x)), y
          if (match(A, m_FNeg(m_Value(A))))
            AddAffected(A);
          if (match(A, m_FAbs(m_Value(A))))
            AddAffected(A);
        }
      }
    } else if (match(V, m_Intrinsic<Intrinsic::is_fpclass>(m_Value(A),
                                                           m_Value()))) {
      // Handle patterns that computeKnownFPClass() support.
      AddAffected(A);
    }
  }
}

} // namespace llvm

#endif // LLVM_ANALYSIS_CONDITIONCACHEUTIL_H
