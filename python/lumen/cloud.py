"""
Gaussian cloud representation and Stanford PLY reader and writer.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Gaussian3D:
    """Single 3D Gaussian Primitive in spatial radiance field."""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: Tuple[float, float, float] = (0.05, 0.05, 0.05)
    rotation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # (w, x, y, z)
    opacity: float = 0.8
    sh_coeffs: List[float] = field(default_factory=lambda: [0.0] * 48)


class GaussianCloud:
    """In-memory collection of 3D Gaussian splat primitives."""

    def __init__(self, splats: Optional[List[Gaussian3D]] = None):
        self.splats: List[Gaussian3D] = splats if splats is not None else []

    def __len__(self) -> int:
        return len(self.splats)

    def add(self, splat: Gaussian3D) -> None:
        """Add a Gaussian primitive to the cloud."""
        self.splats.append(splat)

    def bounding_box(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Compute the axis-aligned bounding box enclosing all Gaussian centers."""
        if not self.splats:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        min_x = min(g.position[0] for g in self.splats)
        min_y = min(g.position[1] for g in self.splats)
        min_z = min(g.position[2] for g in self.splats)

        max_x = max(g.position[0] for g in self.splats)
        max_y = max(g.position[1] for g in self.splats)
        max_z = max(g.position[2] for g in self.splats)

        return ((min_x, min_y, min_z), (max_x, max_y, max_z))

    def centroid(self) -> Tuple[float, float, float]:
        """Compute the arithmetic center of all Gaussian positions."""
        if not self.splats:
            return (0.0, 0.0, 0.0)
        n = float(len(self.splats))
        sx = sum(g.position[0] for g in self.splats)
        sy = sum(g.position[1] for g in self.splats)
        sz = sum(g.position[2] for g in self.splats)
        return (sx / n, sy / n, sz / n)

    @classmethod
    def synthetic_cube(cls, points_per_axis: int = 4, cube_size: float = 1.0) -> "GaussianCloud":
        """Generate a regular 3D lattice cube of colored Gaussian splats."""
        cloud = cls()
        step = cube_size / float(points_per_axis - 1) if points_per_axis > 1 else 0.0
        half = cube_size * 0.5
        splat_scale = (step * 0.6) if step > 0 else 0.1

        c0 = 0.2820947917

        for xi in range(points_per_axis):
            px = -half + float(xi) * step
            for yi in range(points_per_axis):
                py = -half + float(yi) * step
                for zi in range(points_per_axis):
                    pz = -half + float(zi) * step

                    norm_r = (px + half) / cube_size if cube_size > 0 else 0.5
                    norm_g = (py + half) / cube_size if cube_size > 0 else 0.5
                    norm_b = (pz + half) / cube_size if cube_size > 0 else 0.5

                    sh = [0.0] * 48
                    sh[0] = (norm_r - 0.5) / c0
                    sh[1] = (norm_g - 0.5) / c0
                    sh[2] = (norm_b - 0.5) / c0

                    g = Gaussian3D(
                        position=(px, py, pz),
                        scale=(splat_scale, splat_scale, splat_scale),
                        rotation=(1.0, 0.0, 0.0, 0.0),
                        opacity=0.85,
                        sh_coeffs=sh,
                    )
                    cloud.add(g)

        return cloud

    @classmethod
    def synthetic_sphere(cls, point_count: int = 128, radius: float = 1.0) -> "GaussianCloud":
        """Generate a Fibonacci sphere distribution of Gaussian splats."""
        cloud = cls()
        golden_ratio = 1.6180339887
        splat_scale = radius * 0.12
        c0 = 0.2820947917

        for i in range(point_count):
            theta = 2.0 * math.pi * float(i) / golden_ratio
            phi = math.acos(1.0 - 2.0 * (float(i) + 0.5) / float(point_count))

            px = radius * math.sin(phi) * math.cos(theta)
            py = radius * math.sin(phi) * math.sin(theta)
            pz = radius * math.cos(phi)

            norm_r = 0.5 + 0.5 * (px / radius)
            norm_g = 0.5 + 0.5 * (py / radius)
            norm_b = 0.5 + 0.5 * (pz / radius)

            sh = [0.0] * 48
            sh[0] = (norm_r - 0.5) / c0
            sh[1] = (norm_g - 0.5) / c0
            sh[2] = (norm_b - 0.5) / c0

            g = Gaussian3D(
                position=(px, py, pz),
                scale=(splat_scale, splat_scale, splat_scale),
                rotation=(1.0, 0.0, 0.0, 0.0),
                opacity=0.90,
                sh_coeffs=sh,
            )
            cloud.add(g)

        return cloud

    def save_ply(self, filepath: str) -> None:
        """Export Gaussian cloud to ASCII Stanford PLY format."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(self.splats)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float scale_0\n")
            f.write("property float scale_1\n")
            f.write("property float scale_2\n")
            f.write("property float rot_0\n")
            f.write("property float rot_1\n")
            f.write("property float rot_2\n")
            f.write("property float rot_3\n")
            f.write("property float opacity\n")
            f.write("property float f_dc_0\n")
            f.write("property float f_dc_1\n")
            f.write("property float f_dc_2\n")
            f.write("end_header\n")

            for g in self.splats:
                line = (
                    f"{g.position[0]:.6f} {g.position[1]:.6f} {g.position[2]:.6f} "
                    f"{g.scale[0]:.6f} {g.scale[1]:.6f} {g.scale[2]:.6f} "
                    f"{g.rotation[0]:.6f} {g.rotation[1]:.6f} {g.rotation[2]:.6f} {g.rotation[3]:.6f} "
                    f"{g.opacity:.6f} "
                    f"{g.sh_coeffs[0]:.6f} {g.sh_coeffs[1]:.6f} {g.sh_coeffs[2]:.6f}\n"
                )
                f.write(line)

    @classmethod
    def load_ply(cls, filepath: str) -> "GaussianCloud":
        """Load Gaussian cloud from ASCII Stanford PLY file."""
        cloud = cls()
        header_ended = False

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not header_ended:
                    if line_str == "end_header":
                        header_ended = True
                    continue

                if not line_str:
                    continue

                parts = line_str.split()
                if len(parts) >= 14:
                    pos = (float(parts[0]), float(parts[1]), float(parts[2]))
                    scl = (float(parts[3]), float(parts[4]), float(parts[5]))
                    rot = (float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9]))
                    opc = float(parts[10])
                    sh = [0.0] * 48
                    sh[0] = float(parts[11])
                    sh[1] = float(parts[12])
                    sh[2] = float(parts[13])

                    cloud.add(Gaussian3D(
                        position=pos,
                        scale=scl,
                        rotation=rot,
                        opacity=opc,
                        sh_coeffs=sh,
                    ))

        return cloud
