//===-- runtime/engine.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_ENGINE_H_
#define FORTRAN_RUNTIME_ENGINE_H_

#include "derived.h"
#include "stat.h"
#include "terminator.h"
#include "type-info.h"
#include "flang/Runtime/descriptor.h"

namespace Fortran::runtime::engine {

class Engine;

enum class Job { Initialize };

struct CommonState {
  struct Iteration {
    bool Iterating(std::size_t iters, const Descriptor *dtor = nullptr) {
      if (!active) {
        if (iters > 0) {
          active = true;
          at = 0;
          n = iters;
          descriptor = dtor;
          if (descriptor) {
            descriptor->GetLowerBounds(subscripts);
          }
        }
      } else if (resuming) {
        resuming = false;
      } else if (++at < n) {
        if (descriptor) {
          descriptor->IncrementSubscripts(subscripts);
        }
      } else {
        active = false;
      }
      return active;
    }
    void ResumeAtSameIteration() { resuming = true; }

    bool active{false}, resuming{false};
    std::size_t at, n;
    const Descriptor *descriptor;
    SubscriptValue subscripts[maxRank];
  };

  const Descriptor &instance_;
  const typeInfo::DerivedType *derived_;
  const Descriptor *componentDesc_;
  std::size_t elements_, components_;
  Iteration element_, component_;
  StaticDescriptor<maxRank, true, 8> staticDescriptor_;
};

class Initialization : protected CommonState {
public:
  int Run(Engine &); // in derived.cpp
private:
  SubscriptValue extents_[maxRank];
};

class Engine {
public:
  Engine(Terminator &terminator, bool hasStat, const Descriptor *errMsg)
      : terminator_{terminator}, hasStat_{hasStat}, errMsg_{errMsg} {}

  // Start and run a job to completion.
  int Do(Job, const Descriptor &instance, const typeInfo::DerivedType *);

  Terminator &terminator() const { return terminator_; }
  bool hasStat() const { return hasStat_; }
  const Descriptor *errMsg() const { return errMsg_; }

  // Call from running job to suspend execution and start a nested job
  int Begin(Job, const Descriptor &instance, const typeInfo::DerivedType *);
  // Call from a running job to terminate successfully
  int Done();

private:
  class Work {
  public:
    Work(Job job, const Descriptor &instance, const typeInfo::DerivedType *);
    int Run(Engine &); // nonzero on fatal error
  private:
    Job job_;
    union {
      CommonState commonState;
      Initialization initialization;
    } u_;
  };

  int Run();

  Terminator &terminator_;
  bool hasStat_{false};
  const Descriptor *errMsg_;
  int depth_{0};
  static constexpr int maxDepth{4};
  alignas(Work) char workBuf_[maxDepth][sizeof(Work)];
};

} // namespace Fortran::runtime::engine
#endif // FORTRAN_RUNTIME_ENGINE_H_
