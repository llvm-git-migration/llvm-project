//===- unittests/InstallAPI/FileList.cpp - File List Tests ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/InstallAPI/FileList.h"
#include "clang/InstallAPI/HeaderFile.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang::installapi;

namespace FileListTests {

static inline void testValidFileList(std::string Input, HeaderSeq &Expected) {
  auto InputBuf = MemoryBuffer::getMemBuffer(Input);
  HeaderSeq Headers;
  llvm::Error Err = FileListReader::loadHeaders(std::move(InputBuf), Headers);
  EXPECT_FALSE(!!Err);

  EXPECT_EQ(Expected.size(), Headers.size());
  EXPECT_EQ(Headers, Expected);
}

TEST(FileListReader, Version3) {
  static const char Input[] = R"({
    "version" : "3",
    "headers" : [
      {
        "type" : "public",
        "path" : "/tmp/dst/usr/include/foo.h",
        "language" : "objective-c"
      },
      {
        "type" : "private",
        "path" : "/tmp/dst/usr/local/include/bar.h",
        "language" : "objective-c++"
      },
      {
        "type" : "project",
        "path" : "/tmp/src/baz.h"
      }
    ]
  })";

  HeaderSeq Expected = {
      {"/tmp/dst/usr/include/foo.h", HeaderType::Public, "foo.h",
       clang::Language::ObjC},
      {"/tmp/dst/usr/local/include/bar.h", HeaderType::Private, "bar.h",
       clang::Language::ObjCXX},
      {"/tmp/src/baz.h", HeaderType::Project, "", std::nullopt}};

  testValidFileList(Input, Expected);
}

TEST(FileList, Version1) {
  static const char Input[] = R"({
    "version" : "1",
    "headers" : [
      {
        "type" : "public",
        "path" : "/usr/include/foo.h"
      },
      {
        "type" : "private",
        "path" : "/usr/local/include/bar.h"
      }
    ]
  })";

  HeaderSeq Expected = {
      {"/usr/include/foo.h", HeaderType::Public, "foo.h", std::nullopt},
      {"/usr/local/include/bar.h", HeaderType::Private, "bar.h", std::nullopt},
  };

  testValidFileList(Input, Expected);
}

TEST(FileListReader, Version2) {
  static const auto Input = R"({
    "version" : "2",
    "headers" : [
      {
        "type" : "public",
        "path" : "/usr/include/foo.h"
      },
      {
        "type" : "project",
        "path" : "src/bar.h"
      }
    ]
  })";
  HeaderSeq Expected = {
      {"/usr/include/foo.h", HeaderType::Public, "foo.h", std::nullopt},
      {"src/bar.h", HeaderType::Project, "", std::nullopt},
  };

  testValidFileList(Input, Expected);
}

TEST(FileList, MissingVersion) {
  static const char Input[] = R"({
    "headers" : [
      {
        "type" : "public",
        "path" : "/usr/include/foo.h"
      },
      {
        "type" : "private",
        "path" : "/usr/local/include/bar.h"
      }
    ]
  })";
  auto InputBuf = MemoryBuffer::getMemBuffer(Input);
  HeaderSeq Headers;
  llvm::Error Err = FileListReader::loadHeaders(std::move(InputBuf), Headers);
  EXPECT_TRUE(!!Err);
  EXPECT_STREQ("invalid input format: required field 'version' not specified\n",
               toString(std::move(Err)).c_str());
}

TEST(FileList, InvalidTypes) {
  static const char Input[] = R"({
    "version" : "1",
    "headers" : [
      {
        "type" : "project",
        "path" : "/usr/include/foo.h"
      }
    ]
  })";
  auto InputBuf = MemoryBuffer::getMemBuffer(Input);
  HeaderSeq Headers;
  llvm::Error Err = FileListReader::loadHeaders(std::move(InputBuf), Headers);
  EXPECT_TRUE(!!Err);
  EXPECT_STREQ("invalid input format: unsupported header type\n",
               toString(std::move(Err)).c_str());
}
} // namespace FileListTests
