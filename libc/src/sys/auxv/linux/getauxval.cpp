//===-- Implementation file for getauxval function --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/auxv/getauxval.h"
#include "config/linux/app.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/common.h"
#include "src/__support/threads/callonce.h"
#include "src/__support/threads/linux/futex_word.h"
#include "src/errno/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/stdlib/atexit.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munmap.h"
#include "src/sys/prctl/prctl.h"
#include "src/unistd/close.h"
#include "src/unistd/read.h"
#include <linux/auxvec.h>

namespace LIBC_NAMESPACE {

constexpr static size_t AUXV_SIZE = 64;

// Helper to recover or set errno
struct AuxvErrnoGuard {
  int saved;
  bool failure;
  AuxvErrnoGuard() : saved(libc_errno), failure(false) {}
  ~AuxvErrnoGuard() { libc_errno = failure ? ENOENT : saved; }
  void mark_failure() { failure = true; }
};

// Helper to manage the memory
static AuxEntry *auxv = nullptr;

struct AuxvMMapGuard {
  constexpr static size_t AUXV_MMAP_SIZE = sizeof(AuxEntry) * AUXV_SIZE;
  void *ptr;
  AuxvMMapGuard(size_t size)
      : ptr(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, -1, 0)) {}
  ~AuxvMMapGuard() {
    if (ptr != MAP_FAILED) {
      munmap(ptr, AUXV_MMAP_SIZE);
    }
  }
  void submit_to_global() {
    // atexit may fail, we do not set it to global in that case.
    int ret = atexit([]() {
      munmap(auxv, AUXV_MMAP_SIZE);
      auxv = nullptr;
    });

    if (ret != 0)
      return;

    auxv = reinterpret_cast<AuxEntry *>(ptr);
    ptr = MAP_FAILED;
  }
  bool allocated() { return ptr != MAP_FAILED; }
};

struct AuxvFdGuard {
  int fd;
  AuxvFdGuard() : fd(open("/proc/self/auxv", O_RDONLY | O_CLOEXEC)) {}
  ~AuxvFdGuard() {
    if (fd != -1) {
      close(fd);
    }
  }
  bool valid() { return fd != -1; }
};

static void initialize_auxv_once(void) {
  AuxvMMapGuard mmap_guard(AuxvMMapGuard::AUXV_MMAP_SIZE);
  if (!mmap_guard.allocated())
    return;
  auto *ptr = reinterpret_cast<AuxEntry *>(mmap_guard.ptr);
  // we get one less than the max size to make sure the search always
  // terminates.
  size_t available_size = AuxvMMapGuard::AUXV_MMAP_SIZE - sizeof(AuxEntryType);
#if defined(PR_GET_AUXV)
  int ret = prctl(PR_GET_AUXV, reinterpret_cast<unsigned long>(ptr),
                  available_size, 0, 0);
  if (ret >= 0) {
    mmap_guard.submit_to_global();
    return;
  }
#endif
  AuxvFdGuard fd_guard;
  if (!fd_guard.valid())
    return;
  auto *buf = reinterpret_cast<char *>(ptr);
  libc_errno = 0;
  bool error_detected = false;
  while (true) {
    ssize_t bytes_read = read(fd_guard.fd, buf, available_size);
    if (bytes_read <= 0) {
      if (libc_errno == EINTR)
        continue;
      error_detected = bytes_read < 0;
      break;
    }
    available_size -= bytes_read;
  }
  if (!error_detected) {
    mmap_guard.submit_to_global();
  }
}

static AuxEntry read_entry(int fd) {
  AuxEntry buf;
  ssize_t size = sizeof(AuxEntry);
  while (size > 0) {
    ssize_t ret = read(fd, &buf, size);
    if (ret < 0) {
      if (libc_errno == EINTR)
        continue;
      buf.id = AT_NULL;
      buf.value = AT_NULL;
      break;
    }
    size -= ret;
  }
  return buf;
}

LLVM_LIBC_FUNCTION(unsigned long, getauxval, (unsigned long id)) {
  // Fast path when libc is loaded by its own initialization code. In this case,
  // app.auxv_ptr is already set to the auxv passed on the initial stack of the
  // process.
  AuxvErrnoGuard errno_guard;

  auto search_auxv = [&errno_guard](AuxEntry *auxv, unsigned long id) {
    for (auto *ptr = auxv; ptr->id != AT_NULL; ptr++) {
      if (ptr->id == id) {
        return ptr->value;
      }
    }
    errno_guard.mark_failure();
    return AuxEntryType{0};
  };

  if (&app != nullptr) {
    return search_auxv(app.auxv_ptr, id);
  }

  static cpp::Atomic<FutexWordType> once_flag;
  callonce(reinterpret_cast<CallOnceFlag *>(&once_flag), initialize_auxv_once);
  if (auxv != nullptr) {
    return search_auxv(auxv, id);
  }

  // fallback to use read without mmap
  do {
    AuxvFdGuard fd_guard;
    if (!fd_guard.valid())
      break;

    while (true) {
      AuxEntry buf = read_entry(fd_guard.fd);
      if (buf.id == AT_NULL)
        break;
      if (buf.id == id)
        return buf.value;
    }

  } while (0);
  errno_guard.mark_failure();
  return 0;
}
} // namespace LIBC_NAMESPACE
