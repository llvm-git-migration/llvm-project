//===- DependencyScanningFilesystemTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningFilesystem.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gtest/gtest.h"

using namespace clang::tooling::dependencies;

namespace {
 struct InstrumentingInMemoryFilesystem
     : llvm::RTTIExtends<InstrumentingInMemoryFilesystem, llvm::vfs::InMemoryFileSystem> {
   unsigned NumStatCalls = 0;

   using llvm::RTTIExtends<InstrumentingInMemoryFilesystem,
                           llvm::vfs::InMemoryFileSystem>::RTTIExtends;

   llvm::ErrorOr<llvm::vfs::Status> status(const llvm::Twine &Path) override {
     ++NumStatCalls;
     return InMemoryFileSystem::status(Path);
   }
 };
 } // namespace


TEST(DependencyScanningFilesystem, CacheStatOnExists) {
  auto InMemoryInstrumentingFS = llvm::makeIntrusiveRefCnt<InstrumentingInMemoryFilesystem>();
  InMemoryInstrumentingFS->setCurrentWorkingDirectory("/");
  InMemoryInstrumentingFS->addFile("/foo", 0, llvm::MemoryBuffer::getMemBuffer(""));
  InMemoryInstrumentingFS->addFile("/bar", 0, llvm::MemoryBuffer::getMemBuffer(""));
  DependencyScanningFilesystemSharedCache SharedCache;
  DependencyScanningWorkerFilesystem DepFS(SharedCache, InMemoryInstrumentingFS);

  DepFS.status("/foo");
  DepFS.status("/foo");
  DepFS.status("/bar");
  EXPECT_EQ(InMemoryInstrumentingFS->NumStatCalls, 2u);

  DepFS.exists("/foo");
  DepFS.exists("/bar");
  EXPECT_EQ(InMemoryInstrumentingFS->NumStatCalls, 2u);
}
