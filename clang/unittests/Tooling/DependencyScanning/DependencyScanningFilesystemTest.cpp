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
 struct InstrumentingFilesystem
     : llvm::RTTIExtends<InstrumentingFilesystem, llvm::vfs::ProxyFileSystem> {
   unsigned NumStatCalls = 0;

   using llvm::RTTIExtends<InstrumentingFilesystem,
                           llvm::vfs::ProxyFileSystem>::RTTIExtends;

   llvm::ErrorOr<llvm::vfs::Status> status(const llvm::Twine &Path) override {
     ++NumStatCalls;
     return ProxyFileSystem::status(Path);
   }
 };
 } // namespace


TEST(DependencyScanningFilesystem, CacheStatOnExists) {
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS->setCurrentWorkingDirectory("/");
  InMemoryFS->addFile("/foo", 0, llvm::MemoryBuffer::getMemBuffer(""));
  InMemoryFS->addFile("/bar", 0, llvm::MemoryBuffer::getMemBuffer(""));

  auto InstrumentingFS =
    llvm::makeIntrusiveRefCnt<InstrumentingFilesystem>(InMemoryFS);
  DependencyScanningFilesystemSharedCache SharedCache;
  DependencyScanningWorkerFilesystem DepFS(SharedCache, InstrumentingFS);

  DepFS.status("/foo");
  DepFS.status("/foo");
  DepFS.status("/bar");
  EXPECT_EQ(InstrumentingFS->NumStatCalls, 2u);

  DepFS.exists("/foo");
  DepFS.exists("/bar");
  EXPECT_EQ(InstrumentingFS->NumStatCalls, 2u);  
}
