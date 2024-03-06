"""
High-performance contiguous memory buffers and SIMD-aligned slice wrappers.
"""

from std.collections import List
from std.simd import SIMD

struct FloatBuffer:
    """
    Contiguous dynamic array of 32-bit floating point values for image and feature buffers.
    """
    var data: List[Float32]

    def __init__(out self, capacity: Int = 0):
        self.data = List[Float32](capacity=capacity)

    def __init__(out self, size: Int, initial_value: Float32):
        self.data = List[Float32](capacity=size)
        for _ in range(size):
            self.data.append(initial_value)

    def size(self) -> Int:
        return len(self.data)

    def append(mut self, value: Float32):
        self.data.append(value)

    def get(self, index: Int) -> Float32:
        return self.data[index]

    def set(mut self, index: Int, value: Float32):
        self.data[index] = value

    def fill(mut self, value: Float32):
        for i in range(len(self.data)):
            self.data[i] = value

    def zero(mut self):
        self.fill(0.0)

    def min_value(self) -> Float32:
        if len(self.data) == 0:
            return 0.0
        var m = self.data[0]
        for i in range(1, len(self.data)):
            if self.data[i] < m:
                m = self.data[i]
        return m

    def max_value(self) -> Float32:
        if len(self.data) == 0:
            return 0.0
        var m = self.data[0]
        for i in range(1, len(self.data)):
            if self.data[i] > m:
                m = self.data[i]
        return m

    def sum(self) -> Float32:
        var s: Float32 = 0.0
        for i in range(len(self.data)):
            s += self.data[i]
        return s


struct UInt64Buffer:
    """
    Contiguous buffer of 64-bit unsigned integers used for composite radix sort keys.
    """
    var data: List[UInt64]

    def __init__(out self, capacity: Int = 0):
        self.data = List[UInt64](capacity=capacity)

    def __init__(out self, size: Int, initial_value: UInt64):
        self.data = List[UInt64](capacity=size)
        for _ in range(size):
            self.data.append(initial_value)

    def size(self) -> Int:
        return len(self.data)

    def append(mut self, value: UInt64):
        self.data.append(value)

    def get(self, index: Int) -> UInt64:
        return self.data[index]

    def set(mut self, index: Int, value: UInt64):
        self.data[index] = value

    def fill(mut self, value: UInt64):
        for i in range(len(self.data)):
            self.data[i] = value


struct IntBuffer:
    """
    Contiguous buffer of 32-bit signed integers for indices and offsets.
    """
    var data: List[Int]

    def __init__(out self, capacity: Int = 0):
        self.data = List[Int](capacity=capacity)

    def __init__(out self, size: Int, initial_value: Int):
        self.data = List[Int](capacity=size)
        for _ in range(size):
            self.data.append(initial_value)

    def size(self) -> Int:
        return len(self.data)

    def append(mut self, value: Int):
        self.data.append(value)

    def get(self, index: Int) -> Int:
        return self.data[index]

    def set(mut self, index: Int, value: Int):
        self.data[index] = value

    def fill(mut self, value: Int):
        for i in range(len(self.data)):
            self.data[i] = value
