// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=alpha.unix.BlockInCriticalSection \
// RUN:   -std=c++11 \
// RUN:   -analyzer-output text \
// RUN:   -verify %s

void sleep(int x) {}

namespace std {
struct mutex {
  void lock() {}
  void unlock() {}
};
template<typename T>
struct lock_guard {
  lock_guard<T>(std::mutex) {}
  ~lock_guard<T>() {}
};
template<typename T>
struct unique_lock {
  unique_lock<T>(std::mutex) {}
  ~unique_lock<T>() {}
};
template<typename T>
struct not_real_lock {
  not_real_lock<T>(std::mutex) {}
};
} // namespace std

void getc() {}
void fgets() {}
void read() {}
void recv() {}

void pthread_mutex_lock() {}
void pthread_mutex_trylock() {}
void pthread_mutex_unlock() {}

void mtx_lock() {}
void mtx_timedlock() {}
void mtx_trylock() {}
void mtx_unlock() {}

void testBlockInCriticalSectionWithStdMutex() {
  std::mutex m;
  m.lock(); // expected-note 5{{Entering critical section here}}
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  getc(); // expected-warning {{Call to blocking function 'getc' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'getc' inside of critical section}}
  fgets(); // expected-warning {{Call to blocking function 'fgets' inside of critical section}}
           // expected-note@-1 {{Call to blocking function 'fgets' inside of critical section}}
  read(); // expected-warning {{Call to blocking function 'read' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'read' inside of critical section}}
  recv(); // expected-warning {{Call to blocking function 'recv' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'recv' inside of critical section}}
  m.unlock();
}

void testBlockInCriticalSectionWithPthreadMutex() {
  pthread_mutex_lock(); // expected-note 10{{Entering critical section here}}
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  getc(); // expected-warning {{Call to blocking function 'getc' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'getc' inside of critical section}}
  fgets(); // expected-warning {{Call to blocking function 'fgets' inside of critical section}}
           // expected-note@-1 {{Call to blocking function 'fgets' inside of critical section}}
  read(); // expected-warning {{Call to blocking function 'read' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'read' inside of critical section}}
  recv(); // expected-warning {{Call to blocking function 'recv' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'recv' inside of critical section}}
  pthread_mutex_unlock();

  pthread_mutex_trylock(); // expected-note 5{{Entering critical section here}}
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  getc(); // expected-warning {{Call to blocking function 'getc' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'getc' inside of critical section}}
  fgets(); // expected-warning {{Call to blocking function 'fgets' inside of critical section}}
           // expected-note@-1 {{Call to blocking function 'fgets' inside of critical section}}
  read(); // expected-warning {{Call to blocking function 'read' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'read' inside of critical section}}
  recv(); // expected-warning {{Call to blocking function 'recv' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'recv' inside of critical section}}
  pthread_mutex_unlock();
}

void testBlockInCriticalSectionC11Locks() {
  mtx_lock(); // expected-note 15{{Entering critical section here}}
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  getc(); // expected-warning {{Call to blocking function 'getc' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'getc' inside of critical section}}
  fgets(); // expected-warning {{Call to blocking function 'fgets' inside of critical section}}
           // expected-note@-1 {{Call to blocking function 'fgets' inside of critical section}}
  read(); // expected-warning {{Call to blocking function 'read' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'read' inside of critical section}}
  recv(); // expected-warning {{Call to blocking function 'recv' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'recv' inside of critical section}}
  mtx_unlock();

  mtx_timedlock(); // expected-note 10{{Entering critical section here}}
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  getc(); // expected-warning {{Call to blocking function 'getc' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'getc' inside of critical section}}
  fgets(); // expected-warning {{Call to blocking function 'fgets' inside of critical section}}
           // expected-note@-1 {{Call to blocking function 'fgets' inside of critical section}}
  read(); // expected-warning {{Call to blocking function 'read' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'read' inside of critical section}}
  recv(); // expected-warning {{Call to blocking function 'recv' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'recv' inside of critical section}}
  mtx_unlock();

  mtx_trylock(); // expected-note 5{{Entering critical section here}}
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  getc(); // expected-warning {{Call to blocking function 'getc' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'getc' inside of critical section}}
  fgets(); // expected-warning {{Call to blocking function 'fgets' inside of critical section}}
           // expected-note@-1 {{Call to blocking function 'fgets' inside of critical section}}
  read(); // expected-warning {{Call to blocking function 'read' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'read' inside of critical section}}
  recv(); // expected-warning {{Call to blocking function 'recv' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'recv' inside of critical section}}
  mtx_unlock();
}

void testBlockInCriticalSectionWithNestedMutexes() {
  std::mutex m, n, k;
  m.lock(); // expected-note 3{{Entering critical section here}}
  n.lock(); // expected-note 3{{Entering critical section here}}
  k.lock(); // expected-note 3{{Entering critical section here}}
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  k.unlock();
  sleep(5); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  n.unlock();
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  m.unlock();
  sleep(3); // no-warning
}

void f() {
  sleep(1000); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
               // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
}

void testBlockInCriticalSectionInterProcedural() {
  std::mutex m;
  m.lock(); // expected-note {{Entering critical section here}}
  f(); // expected-note {{Calling 'f'}}
  m.unlock();
}

void testBlockInCriticalSectionUnexpectedUnlock() {
  std::mutex m;
  m.unlock();
  sleep(1); // no-warning
  m.lock(); // expected-note {{Entering critical section here}}
  sleep(1); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
}

void testBlockInCriticalSectionLockGuard() {
  std::mutex g_mutex;
  std::not_real_lock<std::mutex> not_real_lock(g_mutex);
  sleep(1); // no-warning

  std::lock_guard<std::mutex> lock(g_mutex); // expected-note {{Entering critical section here}}
  sleep(1); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
}

void testBlockInCriticalSectionLockGuardNested() {
  testBlockInCriticalSectionLockGuard(); // expected-note {{Calling 'testBlockInCriticalSectionLockGuard'}}
  sleep(1); // no-warning
}

void testBlockInCriticalSectionUniqueLock() {
  std::mutex g_mutex;
  std::not_real_lock<std::mutex> not_real_lock(g_mutex);
  sleep(1); // no-warning

  std::unique_lock<std::mutex> lock(g_mutex); // expected-note {{Entering critical section here}}
  sleep(1); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
}

void testBlockInCriticalSectionUniqueLockNested() {
  testBlockInCriticalSectionUniqueLock(); // expected-note {{Calling 'testBlockInCriticalSectionUniqueLock'}}
  sleep(1); // no-warning
}
