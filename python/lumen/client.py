"""
Lumen execution engine and C ABI / native fallback bindings.
"""

import ctypes
import os
import math
import sys
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from .camera import Camera
from .cloud import GaussianCloud, Gaussian3D


@dataclass
class RasterizerConfig:
    """Configuration options for tile-based rasterizer."""
    image_width: int = 400
    image_height: int = 300
    tile_size: int = 16
    near_plane: float = 0.1
    far_plane: float = 50.0
    early_termination_alpha: float = 0.99
    gaussian_radius_sigma: float = 3.0


@dataclass
class RenderResult:
    """Output buffers and performance metrics produced by rasterizer."""
    width: int
    height: int
    rgb_bytes: bytes
    depth_values: List[float]
    alpha_values: List[float]
    stats: Dict[str, Any]


class LumenEngine:
    """High-performance Gaussian Splatting execution engine."""

    def __init__(self, config: Optional[RasterizerConfig] = None):
        self.config = config if config is not None else RasterizerConfig()
        self._c_lib = self._load_c_library()

    def _load_c_library(self) -> Optional[ctypes.CDLL]:
        """Attempt to locate and load liblumen shared library via ctypes."""
        search_dirs = [
            os.path.join(os.path.dirname(__file__), "..", "..", "c_api", "build"),
            os.path.join(os.path.dirname(__file__), "..", "..", "build"),
            "/usr/local/lib",
            ".",
        ]
        lib_names = ["liblumen.dylib", "liblumen.so"]

        for d in search_dirs:
            for name in lib_names:
                full_path = os.path.abspath(os.path.join(d, name))
                if os.path.exists(full_path):
                    try:
                        return ctypes.CDLL(full_path)
                    except OSError:
                        pass
        return None

    def render(self, cloud: GaussianCloud, camera: Camera) -> RenderResult:
        """Render a Gaussian cloud from a camera viewpoint."""
        t_start = time.perf_counter()

        w = self.config.image_width
        h = self.config.image_height
        pixel_count = w * h

        rgb_buf = bytearray(pixel_count * 3)
        depth_buf = [1000.0] * pixel_count
        alpha_buf = [0.0] * pixel_count

        fov_y_rad = math.radians(camera.fov_y_deg)
        focal_y = float(h) / (2.0 * math.tan(fov_y_rad * 0.5))
        focal_x = focal_y

        view_mat = camera.view_matrix_4x4()

        # Project splats
        projected = []
        c0 = 0.2820947917

        for g in cloud.splats:
            gx, gy, gz = g.position
            # Transform to camera space
            vx = view_mat[0][0] * gx + view_mat[0][1] * gy + view_mat[0][2] * gz + view_mat[0][3]
            vy = view_mat[1][0] * gx + view_mat[1][1] * gy + view_mat[1][2] * gz + view_mat[1][3]
            vz = -(view_mat[2][0] * gx + view_mat[2][1] * gy + view_mat[2][2] * gz + view_mat[2][3])

            if vz <= self.config.near_plane or vz >= self.config.far_plane:
                continue

            sx = (vx / vz) * focal_x + float(w) * 0.5
            sy = (vy / vz) * focal_y + float(h) * 0.5

            max_scale = max(g.scale[0], g.scale[1], g.scale[2])
            radius = (max_scale * focal_y / vz) * self.config.gaussian_radius_sigma
            if radius < 1.0:
                radius = 1.0

            if sx + radius < 0 or sx - radius >= w or sy + radius < 0 or sy - radius >= h:
                continue

            var = (radius / self.config.gaussian_radius_sigma) ** 2
            inv_var = 1.0 / max(0.5, var)

            cr = min(1.0, max(0.0, 0.5 + c0 * g.sh_coeffs[0]))
            cg = min(1.0, max(0.0, 0.5 + c0 * g.sh_coeffs[1]))
            cb = min(1.0, max(0.0, 0.5 + c0 * g.sh_coeffs[2]))

            projected.append({
                "sx": sx,
                "sy": sy,
                "depth": vz,
                "radius": radius,
                "inv_var": inv_var,
                "cr": cr,
                "cg": cg,
                "cb": cb,
                "opacity": g.opacity,
            })

        # Sort front to back by view-space depth
        projected.sort(key=lambda item: item["depth"])

        tile_size = self.config.tile_size
        tiles_x = (w + tile_size - 1) // tile_size
        tiles_y = (h + tile_size - 1) // tile_size
        touched_tiles = 0

        for ty in range(tiles_y):
            y_start = ty * tile_size
            y_end = min(y_start + tile_size, h)

            for tx in range(tiles_x):
                x_start = tx * tile_size
                x_end = min(x_start + tile_size, w)

                tile_touched = False

                for p in projected:
                    if (
                        p["sx"] + p["radius"] < x_start
                        or p["sx"] - p["radius"] >= x_end
                        or p["sy"] + p["radius"] < y_start
                        or p["sy"] - p["radius"] >= y_end
                    ):
                        continue

                    tile_touched = True
                    rad_sq = p["radius"] * p["radius"]

                    for py in range(y_start, y_end):
                        dy = float(py) - p["sy"]
                        row_offset = py * w

                        for px in range(x_start, x_end):
                            pidx = row_offset + px
                            cur_a = alpha_buf[pidx]
                            if cur_a >= self.config.early_termination_alpha:
                                continue

                            dx = float(px) - p["sx"]
                            dist_sq = dx * dx + dy * dy
                            if dist_sq > rad_sq:
                                continue

                            power = -0.5 * p["inv_var"] * dist_sq
                            g_weight = math.exp(min(0.0, power))
                            alpha = p["opacity"] * g_weight
                            if alpha < (1.0 / 255.0):
                                continue

                            trans = 1.0 - cur_a
                            w_color = alpha * trans

                            cur_r = float(rgb_buf[pidx * 3 + 0]) / 255.0 + w_color * p["cr"]
                            cur_g = float(rgb_buf[pidx * 3 + 1]) / 255.0 + w_color * p["cg"]
                            cur_b = float(rgb_buf[pidx * 3 + 2]) / 255.0 + w_color * p["cb"]

                            rgb_buf[pidx * 3 + 0] = min(255, int(cur_r * 255.0))
                            rgb_buf[pidx * 3 + 1] = min(255, int(cur_g * 255.0))
                            rgb_buf[pidx * 3 + 2] = min(255, int(cur_b * 255.0))

                            if cur_a < 0.5 and p["depth"] < depth_buf[pidx]:
                                depth_buf[pidx] = p["depth"]

                            alpha_buf[pidx] = cur_a + w_color

                if tile_touched:
                    touched_tiles += 1

        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1000.0

        stats = {
            "total_splats": len(cloud),
            "visible_splats": len(projected),
            "culled_splats": len(cloud) - len(projected),
            "total_tiles": tiles_x * tiles_y,
            "touched_tiles": touched_tiles,
            "elapsed_ms": elapsed_ms,
            "fps": (1000.0 / elapsed_ms) if elapsed_ms > 0 else 1000.0,
        }

        return RenderResult(
            width=w,
            height=h,
            rgb_bytes=bytes(rgb_buf),
            depth_values=depth_buf,
            alpha_values=alpha_buf,
            stats=stats,
        )
