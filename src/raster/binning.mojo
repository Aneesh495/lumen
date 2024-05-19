"""
Gaussian-to-tile binning and contiguous tile range boundary indexing.
"""

from .tile import TileGrid
from .radix_sort import encode_tile_depth_key, decode_tile_id, radix_sort_64bit
from src.core.buffer import UInt64Buffer, IntBuffer
from std.collections import List

struct TileRange(ImplicitlyCopyable):
    var start: Int
    var end: Int

    def __init__(out self):
        self.start = -1
        self.end = -1

    def __init__(out self, start: Int, end: Int):
        self.start = start
        self.end = end

    def is_empty(self) -> Bool:
        return self.start == -1 or self.start >= self.end

    def count(self) -> Int:
        if self.is_empty():
            return 0
        return self.end - self.start


struct TileRangeIndex:
    """
    Direct-mapped O(1) index of contiguous sorted splat ranges for every screen tile.
    """
    var ranges: List[TileRange]

    def __init__(out self, total_tiles: Int):
        self.ranges = List[TileRange](capacity=total_tiles)
        for _ in range(total_tiles):
            self.ranges.append(TileRange(-1, -1))

    def get_range(self, tile_id: Int) -> TileRange:
        if tile_id >= 0 and tile_id < len(self.ranges):
            return self.ranges[tile_id]
        return TileRange(-1, -1)

    def set_range(mut self, tile_id: Int, start: Int, end: Int):
        if tile_id >= 0 and tile_id < len(self.ranges):
            self.ranges[tile_id] = TileRange(start, end)

    @staticmethod
    def build_from_sorted_keys(keys: UInt64Buffer, total_tiles: Int) -> TileRangeIndex:
        var index = TileRangeIndex(total_tiles)
        var n = keys.size()
        if n == 0:
            return index^

        var prev_tile: Int = -1
        for i in range(n):
            var curr_tile = Int(decode_tile_id(keys.get(i)))
            if curr_tile != prev_tile:
                if prev_tile >= 0 and prev_tile < total_tiles:
                    var r = index.get_range(prev_tile)
                    index.set_range(prev_tile, r.start, i)
                if curr_tile < total_tiles:
                    index.set_range(curr_tile, i, n)
                prev_tile = curr_tile

        if prev_tile >= 0 and prev_tile < total_tiles:
            var r = index.get_range(prev_tile)
            index.set_range(prev_tile, r.start, n)

        return index^
