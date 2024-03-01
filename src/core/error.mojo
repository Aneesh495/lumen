"""
Error management, input validation, and invariant assertion utilities for Lumen.
"""

struct StatusCode:
    alias SUCCESS: Int = 0
    alias INVALID_ARGUMENT: Int = 1
    alias OUT_OF_BOUNDS: Int = 2
    alias IO_ERROR: Int = 3
    alias PARSE_ERROR: Int = 4
    alias MATRIX_SINGULAR: Int = 5
    alias MEMORY_ALLOCATION_FAILED: Int = 6
    alias CONVERGENCE_FAILURE: Int = 7
    alias UNKNOWN_ERROR: Int = 99

    @staticmethod
    def description(code: Int) -> String:
        if code == StatusCode.SUCCESS:
            return "Operation succeeded without errors"
        elif code == StatusCode.INVALID_ARGUMENT:
            return "Invalid argument supplied to operation"
        elif code == StatusCode.OUT_OF_BOUNDS:
            return "Array or memory access exceeded bounded range"
        elif code == StatusCode.IO_ERROR:
            return "Input or output subsystem encountered a file or socket failure"
        elif code == StatusCode.PARSE_ERROR:
            return "Data structure or serialized format parsing error"
        elif code == StatusCode.MATRIX_SINGULAR:
            return "Matrix inversion failed due to singularity or vanishing determinant"
        elif code == StatusCode.MEMORY_ALLOCATION_FAILED:
            return "Failed to allocate requested contiguous memory buffer"
        elif code == StatusCode.CONVERGENCE_FAILURE:
            return "Numerical solver failed to converge within allowed iterations"
        else:
            return "Unknown runtime status code"


struct LumenStatus:
    var code: Int
    var message: String

    def __init__(out self, code: Int = StatusCode.SUCCESS, message: String = ""):
        self.code = code
        self.message = message

    def is_ok(self) -> Bool:
        return self.code == StatusCode.SUCCESS

    def is_error(self) -> Bool:
        return self.code != StatusCode.SUCCESS

    @staticmethod
    def ok() -> LumenStatus:
        return LumenStatus(StatusCode.SUCCESS, "OK")

    @staticmethod
    def error(code: Int, message: String) -> LumenStatus:
        return LumenStatus(code, message)


def validate_dimension(dim: Int, name: String) -> LumenStatus:
    if dim <= 0:
        return LumenStatus.error(
            StatusCode.INVALID_ARGUMENT,
            "Dimension " + name + " must be strictly positive"
        )
    return LumenStatus.ok()


def validate_range(val: Float32, min_val: Float32, max_val: Float32, name: String) -> LumenStatus:
    if val < min_val or val > max_val:
        return LumenStatus.error(
            StatusCode.OUT_OF_BOUNDS,
            "Value " + name + " is outside permitted range"
        )
    return LumenStatus.ok()
