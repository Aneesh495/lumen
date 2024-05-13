"""
Tile-based screen decomposition and 16x16 pixel block coordinate mapping.
"""

from src.core.types import TILE_WIDTH, TILE_HEIGHT, TILE_PIXELS
from std.math import max, min

struct TileGrid(ImplicitlyCopyable):
    """
    Manages 2D screen subdivision into uniform 16x16 pixel workgroups for parallel rasterization.
    """
    var width: Int
    var height: Int
    var tile_width: Int
    var tile_height: Int
    var tiles_x: Int
    var tiles_y: Int
    var total_tiles: Int

    def __init__(out self, width: Int, height: Int, tile_w: Int = TILE_WIDTH, tile_h: Int = TILE_HEIGHT):
        self.width = width
        self.height = height
        self.tile_width = tile_w
        self.tile_height = tile_h
        self.tiles_x = (width + tile_w - 1) // tile_w
        self.tiles_y = (height + tile_h - 1) // tile_h
        self.total_tiles = self.tiles_x * self.tiles_y

    def get_tile_id(self, tx: Int, ty: Int) -> Int:
        return ty * self.tiles_x + tx

    def get_tile_coord_x(self, tile_id: Int) -> Int:
        return tile_id % self.tiles_x

    def get_tile_coord_y(self, tile_id: Int) -> Int:
        return tile_id // self.tiles_x

    def get_tile_pixel_min_x(self, tx: Int) -> Int:
        return tx * self.tile_width

    def get_tile_pixel_min_y(self, ty: Int) -> Int:
        return ty * self.tile_height

    def get_tile_pixel_max_x(self, tx: Int) -> Int:
        var mx = (tx + 1) * self.tile_width
        if mx > self.width:
            return self.width
        return mx

    def get_tile_pixel_max_y(self, ty: Int) -> Int:
        var my = (ty + 1) * self.tile_height
        if my > self.height:
            return self.height
        return my

    def pixel_to_tile_x(self, px: Int) -> Int:
        return px // self.tile_width

    def pixel_to_tile_y(self, py: Int) -> Int:
        return py // self.tile_height

    def get_gaussian_tile_overlap(
        self,
        center_x: Float32,
        center_y: Float32,
        radius: Int,
        mut out_min_tx: Int,
        mut out_min_ty: Int,
        mut out_max_tx: Int,
        mut out_max_ty: Int
    ) -> Bool:
        """
        Determine bounding range of tiles touched by a 2D Gaussian ellipse of given radius.
        Returns False if the Gaussian is completely off-screen.
        """
        var r_f = Float32(radius)
        var min_px = Int(center_x - r_f)
        var max_px = Int(center_x + r_f)
        var min_py = Int(center_y - r_f)
        var max_py = Int(center_y + r_f)

        # Off-screen test
        if max_px < 0 or min_px >= self.width or max_py < 0 or min_py >= self.height:
            return False

        # Clamp pixel bounds to screen
        if min_px < 0: min_px = 0
        if max_px >= self.width: max_px = self.width - 1
        if min_py < 0: min_py = 0
        if max_py >= self.height: max_py = self.height - 1

        out_min_tx = self.pixel_to_tile_x(min_px)
        out_max_tx = self.pixel_to_tile_x(max_px)
        out_min_ty = self.pixel_to_tile_y(min_py)
        out_max_ty = self.pixel_to_tile_y(max_py)

        return True
