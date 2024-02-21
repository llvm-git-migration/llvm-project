#include <cassert>
#include <shared_recursive_mutex>

constexpr uint32_t SHARED_USER_MAX_CNT = UINT32_MAX;
constexpr pthread_t THREAD_ID_NOEXIST  = 0;

using namespace std;

#ifndef likely
#  define likely(condition) __builtin_expect(!!(condition), 1)
#endif
#ifndef unlikely
#  define unlikely(condition) __builtin_expect(!!(condition), 0)
#endif

void shared_recursive_mutex::lock() {
  pthread_t self = pthread_self();
  std::unique_lock guard(mutex);

  // write lock reentrant
  if (owner == self) {
    if (unlikely(recursions == SHARED_USER_MAX_CNT)) {
      return;
    }
    ++recursions;
    return;
  }

  while ((owner != THREAD_ID_NOEXIST) || (readers != 0)) {
    cond.wait(guard);
  }

  owner      = self;
  recursions = 1;
}

bool shared_recursive_mutex::try_lock() {
  pthread_t self = pthread_self();
  std::lock_guard guard(mutex);

  // write lock reentrant
  if (owner == self) {
    if (unlikely(recursions == SHARED_USER_MAX_CNT)) {
      return false;
    }
    ++recursions;
    return true;
  }

  if ((owner != THREAD_ID_NOEXIST) || (readers != 0)) {
    return false;
  }

  owner      = self;
  recursions = 1;

  return true;
}

void shared_recursive_mutex::unlock() {
  pthread_t self = pthread_self();
  std::lock_guard guard(mutex);

  if (unlikely((owner != self) || (recursions == 0))) {
    return;
  }

  if (--recursions == 0) {
    owner = 0;
    // release write lock and notifies all the servers.
    cond.notify_all();
  }
}

void shared_recursive_mutex::lock_shared() {
  pthread_t self = pthread_self();
  std::unique_lock guard(mutex);

  // write-read nesting
  if (owner == self) {
    ++readers;
    return;
  }

  // If other threads have held the write lock or the number of read locks exceeds the upper limit, wait.
  while (unlikely(owner != THREAD_ID_NOEXIST) || unlikely(readers == SHARED_USER_MAX_CNT)) {
    cond.wait(guard);
  }

  ++readers;
}

bool shared_recursive_mutex::try_lock_shared() {
  pthread_t self = pthread_self();
  std::lock_guard guard(mutex);

  // write-read nesting
  if (owner == self) {
    ++readers;
    return true;
  }

  // If another thread already holds the write lock or the number of read locks exceeds the upper limit, the operation
  // fails.
  if (unlikely(owner != THREAD_ID_NOEXIST) || unlikely(readers == SHARED_USER_MAX_CNT)) {
    return false;
  }

  ++readers;
  return true;
}

void shared_recursive_mutex::unlock_shared() {
  std::lock_guard guard(mutex);

  if (readers == 0) {
    return;
  }

  --readers;
  cond.notify_all();
}
