//===-- runtime/engine.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "engine.h"

namespace Fortran::runtime::engine {

Work::Work(
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

int Work::Run(Engine &engine) {
  switch (job_) {
  case Job::Initialize:
    return u_.initialization.Run(engine);
  }
  engine.terminator().Crash(
      "Work::Run: bad job_ code %d", static_cast<int>(job_));
}

int Engine::Do(
    Job job, const Descriptor &instance, const typeInfo::DerivedType *derived) {
  if (int status{Begin(job, instance, derived)}; status != StatOk) {
    return status;
  }
  return Run();
}

int Engine::Begin(
    Job job, const Descriptor &instance, const typeInfo::DerivedType *derived) {
  // TODO: heap allocation on overflow
  new (workBuf_[depth_++]) Work{job, instance, derived};
  return StatOk;
}

int Engine::Run() {
  while (depth_) {
    auto *w{reinterpret_cast<Work *>(workBuf_[depth_ - 1])};
    if (int status{w->Run(*this)}; status != StatOk) {
      return status;
    }
  }
  return StatOk;
}

int Engine::Done() {
  --depth_;
  return StatOk;
}

} // namespace Fortran::runtime::engine
