# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.func as func
import mlir.dialects.memref as memref
from mlir.dialects.memref import _infer_memref_subview_result_type
import mlir.dialects.arith as arith
import mlir.extras.types as T


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: testSubViewAccessors
@run
def testSubViewAccessors():
    ctx = Context()
    module = Module.parse(
        r"""
    func.func @f1(%arg0: memref<?x?xf32>) {
      %0 = arith.constant 0 : index
      %1 = arith.constant 1 : index
      %2 = arith.constant 2 : index
      %3 = arith.constant 3 : index
      %4 = arith.constant 4 : index
      %5 = arith.constant 5 : index
      memref.subview %arg0[%0, %1][%2, %3][%4, %5] : memref<?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      return
    }
  """,
        ctx,
    )
    func_body = module.body.operations[0].regions[0].blocks[0]
    subview = func_body.operations[6]

    assert subview.source == subview.operands[0]
    assert len(subview.offsets) == 2
    assert len(subview.sizes) == 2
    assert len(subview.strides) == 2
    assert subview.result == subview.results[0]

    # CHECK: SubViewOp
    print(type(subview).__name__)

    # CHECK: constant 0
    print(subview.offsets[0])
    # CHECK: constant 1
    print(subview.offsets[1])
    # CHECK: constant 2
    print(subview.sizes[0])
    # CHECK: constant 3
    print(subview.sizes[1])
    # CHECK: constant 4
    print(subview.strides[0])
    # CHECK: constant 5
    print(subview.strides[1])


# CHECK-LABEL: TEST: testCustomBuidlers
@run
def testCustomBuidlers():
    with Context() as ctx, Location.unknown(ctx):
        module = Module.parse(
            r"""
      func.func @f1(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index) {
        return
      }
    """
        )
        f = module.body.operations[0]
        func_body = f.regions[0].blocks[0]
        with InsertionPoint.at_block_terminator(func_body):
            memref.LoadOp(f.arguments[0], f.arguments[1:])

        # CHECK: func @f1(%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
        # CHECK: memref.load %[[ARG0]][%[[ARG1]], %[[ARG2]]]
        print(module)
        assert module.operation.verify()


# CHECK-LABEL: TEST: testMemRefAttr
@run
def testMemRefAttr():
    with Context() as ctx, Location.unknown(ctx):
        module = Module.create()
        with InsertionPoint(module.body):
            memref.global_("objFifo_in0", T.memref(16, T.i32()))
        # CHECK: memref.global @objFifo_in0 : memref<16xi32>
        print(module)


# CHECK-LABEL: TEST: testSubViewOpInferReturnType
@run
def testSubViewOpInferReturnType():
    with Context() as ctx, Location.unknown(ctx):
        module = Module.create()
        with InsertionPoint(module.body):
            x = memref.alloc(T.memref(10, 10, T.i32()), [], [])
            # CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<10x10xi32>
            print(x.owner)

            y = memref.subview(x, [1, 1], [3, 3], [1, 1])
            # CHECK: %{{.*}} = memref.subview %[[ALLOC]][1, 1] [3, 3] [1, 1] : memref<10x10xi32> to memref<3x3xi32, strided<[10, 1], offset: 11>>
            print(y.owner)

            z = memref.subview(
                x,
                [arith.constant(T.index(), 1), 1],
                [3, 3],
                [1, 1],
            )
            # CHECK: %{{.*}} =  memref.subview %[[ALLOC]][1, 1] [3, 3] [1, 1] : memref<10x10xi32> to memref<3x3xi32, strided<[10, 1], offset: 11>>
            print(z.owner)

            z = memref.subview(
                x,
                [arith.constant(T.index(), 3), arith.constant(T.index(), 4)],
                [3, 3],
                [1, 1],
            )
            # CHECK: %{{.*}} =  memref.subview %[[ALLOC]][3, 4] [3, 3] [1, 1] : memref<10x10xi32> to memref<3x3xi32, strided<[10, 1], offset: 34>>
            print(z.owner)

            try:
                memref.subview(
                    x,
                    [
                        arith.addi(
                            arith.constant(T.index(), 3), arith.constant(T.index(), 4)
                        ),
                        0,
                    ],
                    [3, 3],
                    [1, 1],
                )
            except AssertionError as e:
                # CHECK: mixed static/dynamic offset/sizes/strides requires explicit result type
                print(e)

            try:
                _infer_memref_subview_result_type(
                    x.type,
                    [arith.constant(T.index(), 3), arith.constant(T.index(), 4)],
                    [ShapedType.get_dynamic_size(), 3],
                    [1, 1],
                )
            except AssertionError as e:
                # CHECK: Only inferring from python or mlir integer constant is supported
                print(e)

            try:
                memref.subview(
                    x,
                    [arith.constant(T.index(), 3), arith.constant(T.index(), 4)],
                    [ShapedType.get_dynamic_size(), 3],
                    [1, 1],
                )
            except AssertionError as e:
                # CHECK: mixed static/dynamic offset/sizes/strides requires explicit result type
                print(e)

            layout = StridedLayoutAttr.get(ShapedType.get_dynamic_size(), [10, 1])
            x = memref.alloc(
                T.memref(
                    10,
                    10,
                    T.i32(),
                    layout=layout,
                ),
                [],
                [arith.constant(T.index(), 42)],
            )
            # CHECK: %[[DYNAMICALLOC:.*]] = memref.alloc()[%c42] : memref<10x10xi32, strided<[10, 1], offset: ?>>
            print(x.owner)
            y = memref.subview(
                x,
                [1, 1],
                [3, 3],
                [1, 1],
                result_type=T.memref(3, 3, T.i32(), layout=layout),
            )
            # CHECK: %subview_9 = memref.subview %[[DYNAMICALLOC]][1, 1] [3, 3] [1, 1] : memref<10x10xi32, strided<[10, 1], offset: ?>> to memref<3x3xi32, strided<[10, 1], offset: ?>>
            print(y.owner)
