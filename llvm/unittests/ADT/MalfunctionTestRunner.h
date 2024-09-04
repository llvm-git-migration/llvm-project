//===- llvm/unittest/ADT/MalfunctionTestRunner.h - Test runner for malfunction
// testing --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test runner for OOM scenario tests
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_ADT_MALFUNCTIONTESTRUNNER_H
#define LLVM_UNITTESTS_ADT_MALFUNCTIONTESTRUNNER_H

#ifdef LLVM_EH_ENABLED

#include <iomanip>
#include <ostream>

namespace llvm {

// Interface for a scenario to be tested for malfunctions
class IMalfunctionTestScenario {
public:
  virtual ~IMalfunctionTestScenario() = default;
  // is called before execute, but without malfunctions, used to prepare stuff
  virtual void prepare() {}
  // execute the scenario
  virtual void execute() = 0;
};

// Malfunction test runner
class MalfunctionTestRunner {
private:
  IMalfunctionTestScenario *m_scenario;
  std::ostream &m_log;
  static constexpr uint32_t MaxSuccessfulRuns = 3;

public:
  MalfunctionTestRunner(IMalfunctionTestScenario *Scenario, std::ostream &Log)
      : m_scenario(Scenario), m_log(Log) {}

  void run() {
    m_scenario->prepare();

    uint32_t SuccessfulRuns = 0;
    uint32_t Counter = 0u;
    while (true) {
      m_log << std::setw(3) << Counter;
      m_log.flush(); // need to know the current counter in case of a crash

      bool FinishedWithSuccess = false;
      try {
        m_scenario->execute();
        FinishedWithSuccess = true;
        m_log << ":ok, ";
      } catch (const std::bad_alloc &) {
        m_log << ":ba, ";
      }

      // Insert new line each 10th iteration
      if (Counter % 10 == 0) {
        m_log << std::endl;
      }

      if (FinishedWithSuccess) {
        SuccessfulRuns++;
      } else {
        // ignore successful runs followed by failures
        SuccessfulRuns = 0;
      }

      // Move to the next allocation
      Counter++;

      if (SuccessfulRuns >= MaxSuccessfulRuns)
        break;
    }

    m_log << std::endl << std::endl;
  }
};

} // namespace llvm

#endif // LLVM_EH_ENABLED
#endif // LLVM_UNITTESTS_ADT_MALFUNCTIONTESTRUNNER_H
