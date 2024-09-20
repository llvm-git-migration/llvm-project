import concurrent.futures
import threading
import inspect


def decorator(f):
    # Introspect the callable for optional features.
    sig = inspect.signature(f)
    for param in sig.parameters.values():
        pass

    def emit_call_op(*call_args):
        pass

    wrapped = emit_call_op
    return wrapped


def test_dialects_vector_repro_3():
    num_workers = 6
    num_runs = 10
    barrier = threading.Barrier(num_workers)

    def closure():
        barrier.wait()
        for _ in range(num_runs):

            @decorator
            def print_vector(arg):
                return 0

        barrier.wait()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:
        futures = []
        for _ in range(num_workers):
            futures.append(executor.submit(closure))
        # We should call future.result() to re-raise an exception if test has
        # failed
        assert len(list(f.result() for f in futures)) == num_workers


if __name__ == "__main__":
    test_dialects_vector_repro_3()