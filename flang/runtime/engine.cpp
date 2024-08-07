//===-- runtime/engine.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "engine.h"

namespace Fortran::runtime::engine {

Engine::Work::Work(
    Job job, const Descriptor &instance, const typeInfo::DerivedType *derived)
    : job_{job}, u_{{instance}} {
  auto &state{u_.commonState};
  state.derived_ = derived;
  state.elements_ = instance.Elements();
  if (derived) {
    state.componentDesc_ = &derived->component();
    state.components_ = state.componentDesc_->Elements();
  } else {
    state.componentDesc_ = nullptr;
    state.components_ = 0;
  }
}

void Engine::Work::Resume(Engine &engine) {
  switch (job_) {
  case Job::Initialization:
    u_.initialization.Resume(engine);
    return;
  }
  engine.terminator().Crash(
      "Work::Run: bad job_ code %d", static_cast<int>(job_));
}

int Engine::Do(
    Job job, const Descriptor &instance, const typeInfo::DerivedType *derived) {
  Begin(job, instance, derived);
  while (depth_ > 0) {
    if (status_ == StatOk) {
      auto *w{reinterpret_cast<Work *>(workBuf_[depth_ - 1])};
      w->Resume(*this);
    } else {
      Done();
    }
  }
  return status_;
}

Task::ResultType Engine::Begin(
    Job job, const Descriptor &instance, const typeInfo::DerivedType *derived) {
  // TODO: heap allocation on overflow
  RUNTIME_CHECK(terminator_, depth_ < maxDepth);
  new (workBuf_[depth_++]) Work{job, instance, derived};
  return Task::ResultType::ResultValue;
}

Task::ResultType Engine::Done() {
  --depth_;
  return Task::ResultType::ResultValue;
}

Task::ResultType Engine::Fail(int status) {
  status_ = status;
  return Done();
}

} // namespace Fortran::runtime::engine
