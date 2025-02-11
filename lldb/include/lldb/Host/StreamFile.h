//===-- StreamFile.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_STREAMFILE_H
#define LLDB_HOST_STREAMFILE_H

#include "lldb/Host/File.h"
#include "lldb/Utility/Stream.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"

#include <cstdint>
#include <cstdio>

namespace lldb_private {

class StreamFile : public Stream {
public:
  StreamFile(uint32_t flags, uint32_t addr_size, lldb::ByteOrder byte_order);

  StreamFile(int fd, bool transfer_ownership);

  StreamFile(const char *path, File::OpenOptions options,
             uint32_t permissions = lldb::eFilePermissionsFileDefault);

  StreamFile(FILE *fh, bool transfer_ownership);

  StreamFile(std::shared_ptr<File> file) : m_file_sp(file) { assert(file); };

  ~StreamFile() override;

  File &GetFile() { return *m_file_sp; }

  const File &GetFile() const { return *m_file_sp; }

  std::shared_ptr<File> GetFileSP() { return m_file_sp; }

  void Flush() override;

protected:
  // Classes that inherit from StreamFile can see and modify these
  std::shared_ptr<File> m_file_sp; // never NULL
  size_t WriteImpl(const void *s, size_t length) override;

private:
  StreamFile(const StreamFile &) = delete;
  const StreamFile &operator=(const StreamFile &) = delete;
};

class SynchronizedStreamFile : public StreamFile {
public:
  SynchronizedStreamFile(uint32_t flags, uint32_t addr_size,
                         lldb::ByteOrder byte_order)
      : StreamFile(flags, addr_size, byte_order) {}

  SynchronizedStreamFile(int fd, bool transfer_ownership)
      : StreamFile(fd, transfer_ownership) {}

  SynchronizedStreamFile(
      const char *path, File::OpenOptions options,
      uint32_t permissions = lldb::eFilePermissionsFileDefault)
      : StreamFile(path, options, permissions) {}

  SynchronizedStreamFile(FILE *fh, bool transfer_ownership)
      : StreamFile(fh, transfer_ownership) {}

  SynchronizedStreamFile(std::shared_ptr<File> file) : StreamFile(file) {}

  ~SynchronizedStreamFile() override;

  std::recursive_mutex &GetMutex() { return m_mutex; }

protected:
  size_t WriteImpl(const void *s, size_t length) override;
  std::recursive_mutex m_mutex;

private:
  SynchronizedStreamFile(const SynchronizedStreamFile &) = delete;
  const SynchronizedStreamFile &
  operator=(const SynchronizedStreamFile &) = delete;
};

} // namespace lldb_private

#endif // LLDB_HOST_STREAMFILE_H
