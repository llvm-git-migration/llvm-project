#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#ifdef _WIN32
#include "Windows.h"

int getpagesize() {
  SYSTEM_INFO system_info;
  GetSystemInfo(&system_info);
  return system_info.dwPageSize;
}

void *allocate_memory_with_holes() {
  int pagesize = getpagesize();
  void *mem = VirtualAlloc(nullptr, 5 * pagesize, MEM_RESERVE, PAGE_NOACCESS);
  if (!mem) {
    std::cerr << std::system_category().message(GetLastError()) << std::endl;
    exit(1);
  }
  char *bytes = static_cast<char *>(mem);
  for (int page : {0, 2, 4}) {
    if (!VirtualAlloc(bytes + page * pagesize, pagesize, MEM_COMMIT,
                      PAGE_READWRITE)) {
      std::cerr << std::system_category().message(GetLastError()) << std::endl;
      exit(1);
    }
  }
  return bytes;
}
#else
#include "sys/mman.h"
#include "unistd.h"

char *allocate_memory_with_holes() {
  int pagesize = getpagesize();
  void *mem = mmap(nullptr, 5 * pagesize, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mem == MAP_FAILED) {
    perror("mmap");
    exit(1);
  }
  char *bytes = static_cast<char *>(mem);
  for (int page : {1, 3}) {
    if (munmap(bytes + page * pagesize, pagesize) != 0) {
      perror("munmap");
      exit(1);
    }
  }
  return bytes;
}
#endif

int main(int argc, char const *argv[]) {
  const char *stringdata =
      "hello world; I like to write text in const char pointers";
  uint8_t bytedata[] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11,
                        0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99};

  char *mem_with_holes = allocate_memory_with_holes();
  int pagesize = getpagesize();
  char *matches[] = {
      mem_with_holes,                // Beginning of memory
      mem_with_holes + 2 * pagesize, // After a hole
      mem_with_holes + 2 * pagesize +
          pagesize / 2, // Middle of a block, after an existing match.
      mem_with_holes + 5 * pagesize - 7, // End of memory
  };
  for (char *m : matches)
    strcpy(m, "needle");

  return 0; // break here
}
