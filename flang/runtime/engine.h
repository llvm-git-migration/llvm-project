//===-- runtime/engine.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements a work engine for restartable tasks iterating over elements,
// components, &c. of arrays and derived types.  Avoids recursion and
// function pointers.

#ifndef FORTRAN_RUNTIME_ENGINE_H_
#define FORTRAN_RUNTIME_ENGINE_H_

#include "derived.h"
#include "stat.h"
#include "terminator.h"
#include "type-info.h"
#include "flang/Runtime/descriptor.h"

namespace Fortran::runtime::engine {

class Engine;

// Every task object derives from Task.
struct Task {

  enum class ResultType { ResultValue /*doesn't matter*/ };

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
    // Call on all Iteration instances before calling Engine::Begin()
    // when they should not advance when the job is resumed.
    void ResumeAtSameIteration() { resuming = true; }

    bool active{false}, resuming{false};
    std::size_t at, n;
    const Descriptor *descriptor;
    SubscriptValue subscripts[maxRank];
  };

  // For looping over elements
  const Descriptor &instance_;
  std::size_t elements_;
  Iteration element_;

  // For looping over components
  const typeInfo::DerivedType *derived_;
  const Descriptor *componentDesc_;
  std::size_t components_;
  Iteration component_;
};

enum class Job { Initialization };

class Initialization : protected Task {
public:
  ResultType Resume(Engine &);

private:
  SubscriptValue extents_[maxRank];
  StaticDescriptor<maxRank, true, 8> staticDescriptor_;
};

class Engine {
public:
  Engine(Terminator &terminator, bool hasStat, const Descriptor *errMsg)
      : terminator_{terminator}, hasStat_{hasStat}, errMsg_{errMsg} {}

  Terminator &terminator() const { return terminator_; }
  bool hasStat() const { return hasStat_; }
  const Descriptor *errMsg() const { return errMsg_; }

  // Start and run a job to completion; returns status code.
  int Do(Job, const Descriptor &instance, const typeInfo::DerivedType *);

  // Callbacks from running tasks for use in their return statements.
  // Suspends execution and start a nested job
  Task::ResultType Begin(
      Job, const Descriptor &instance, const typeInfo::DerivedType *);
  // Terminates task successfully
  Task::ResultType Done();
  // Terminates task unsuccessfully
  Task::ResultType Fail(int status);

private:
  class Work {
  public:
    Work(Job job, const Descriptor &instance, const typeInfo::DerivedType *);
    void Resume(Engine &);

  private:
    Job job_;
    union {
      Task commonState;
      Initialization initialization;
    } u_;
  };

  Terminator &terminator_;
  bool hasStat_{false};
  const Descriptor *errMsg_;
  int status_{StatOk};
  int depth_{0};
  static constexpr int maxDepth{4};
  alignas(Work) char workBuf_[maxDepth][sizeof(Work)];
};

} // namespace Fortran::runtime::engine
#endif // FORTRAN_RUNTIME_ENGINE_H_
