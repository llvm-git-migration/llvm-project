//===- unittests/InstallAPI/FileList.cpp - File List Tests ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/InstallAPI/FileList.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"
#include <map>

using namespace llvm;
using namespace clang::installapi;

namespace {
class TestVisitor : public FileListReader::Visitor {
public:
  std::map<StringRef, FileListReader::HeaderInfo> Headers;

  void visitHeaderFile(FileListReader::HeaderInfo &Header) override {
    StringRef Key = StringRef(Header.Path).rsplit("/").second;
    Headers[Key] = Header;
  }
};
} // namespace

static bool operator==(const FileListReader::HeaderInfo &LHS,
                       const FileListReader::HeaderInfo &RHS) {
  return std::tie(LHS.Type, LHS.Path, LHS.Language) ==
         std::tie(RHS.Type, RHS.Path, RHS.Language);
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
  auto InputBuf = MemoryBuffer::getMemBuffer(Input);
  auto Reader = FileListReader::get(std::move(InputBuf));
  ASSERT_TRUE(!!Reader);
  TestVisitor Visitor;
  (*Reader)->visit(Visitor);
  EXPECT_EQ(3U, Visitor.Headers.size());

  FileListReader::HeaderInfo Foo{
      HeaderType::Public, "/tmp/dst/usr/include/foo.h", clang::Language::ObjC};
  EXPECT_TRUE(Foo == Visitor.Headers["foo.h"]);

  FileListReader::HeaderInfo Bar{HeaderType::Private,
                                 "/tmp/dst/usr/local/include/bar.h",
                                 clang::Language::ObjCXX};
  EXPECT_TRUE(Bar == Visitor.Headers["bar.h"]);

  FileListReader::HeaderInfo Baz{HeaderType::Project, "/tmp/src/baz.h",
                                 std::nullopt};
  EXPECT_TRUE(Baz == Visitor.Headers["baz.h"]);
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
  auto InputBuf = MemoryBuffer::getMemBuffer(Input);
  auto Reader = FileListReader::get(std::move(InputBuf));
  ASSERT_TRUE(!!Reader);

  TestVisitor Visitor;
  (*Reader)->visit(Visitor);
  EXPECT_EQ(2U, Visitor.Headers.size());

  FileListReader::HeaderInfo Foo{HeaderType::Public, "/usr/include/foo.h",
                                 std::nullopt};
  EXPECT_TRUE(Foo == Visitor.Headers["foo.h"]);

  FileListReader::HeaderInfo Bar{HeaderType::Private,
                                 "/usr/local/include/bar.h", std::nullopt};
  EXPECT_TRUE(Bar == Visitor.Headers["bar.h"]);
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
  auto Reader = FileListReader::get(std::move(InputBuf));
  EXPECT_FALSE(!!Reader);
  consumeError(Reader.takeError());
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
  auto Reader = FileListReader::get(std::move(InputBuf));
  EXPECT_FALSE(!!Reader);
  EXPECT_STREQ("invalid input format: unsupported header type\n",
               toString(Reader.takeError()).c_str());
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
  auto InputBuf = MemoryBuffer::getMemBuffer(Input);
  auto Reader = FileListReader::get(std::move(InputBuf));
  ASSERT_TRUE(!!Reader);

  TestVisitor Visitor;
  (*Reader)->visit(Visitor);
  EXPECT_EQ(2U, Visitor.Headers.size());

  FileListReader::HeaderInfo Foo{HeaderType::Public, "/usr/include/foo.h",
                                 std::nullopt};
  EXPECT_TRUE(Foo == Visitor.Headers["foo.h"]);

  FileListReader::HeaderInfo Bar{HeaderType::Project, "src/bar.h",
                                 std::nullopt};
  EXPECT_TRUE(Bar == Visitor.Headers["bar.h"]);
}
