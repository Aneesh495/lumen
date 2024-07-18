"""
Camera representation and trajectory generators for Lumen radiance field rendering.
"""

import math
from typing import List, Tuple


class Camera:
    """Pinhole camera descriptor for rendering 3D Gaussian radiance fields."""

    def __init__(
        self,
        position: Tuple[float, float, float] = (0.0, 0.0, 3.0),
        target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        fov_x_deg: float = 60.0,
        fov_y_deg: float = 45.0,
        near_plane: float = 0.1,
        far_plane: float = 100.0,
    ):
        self.position = list(position)
        self.target = list(target)
        self.up = list(up)
        self.fov_x_deg = float(fov_x_deg)
        self.fov_y_deg = float(fov_y_deg)
        self.near_plane = float(near_plane)
        self.far_plane = float(far_plane)

    def forward_vector(self) -> Tuple[float, float, float]:
        """Compute forward unit direction from position to target."""
        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]
        dz = self.target[2] - self.position[2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-6:
            return (0.0, 0.0, -1.0)
        return (dx / length, dy / length, dz / length)

    def view_matrix_4x4(self) -> List[List[float]]:
        """Compute 4x4 view transform matrix (look-at)."""
        f = self.forward_vector()

        # right = normalize(cross(f, up))
        rx = f[1] * self.up[2] - f[2] * self.up[1]
        ry = f[2] * self.up[0] - f[0] * self.up[2]
        rz = f[0] * self.up[1] - f[1] * self.up[0]
        r_len = math.sqrt(rx * rx + ry * ry + rz * rz)
        if r_len < 1e-6:
            rx, ry, rz = (1.0, 0.0, 0.0)
        else:
            rx, ry, rz = rx / r_len, ry / r_len, rz / r_len

        # true up = cross(r, f)
        ux = ry * f[2] - rz * f[1]
        uy = rz * f[0] - rx * f[2]
        uz = rx * f[1] - ry * f[0]

        # translation dot products
        tx = -(rx * self.position[0] + ry * self.position[1] + rz * self.position[2])
        ty = -(ux * self.position[0] + uy * self.position[1] + uz * self.position[2])
        tz = f[0] * self.position[0] + f[1] * self.position[1] + f[2] * self.position[2]

        return [
            [rx, ry, rz, tx],
            [ux, uy, uz, ty],
            [-f[0], -f[1], -f[2], tz],
            [0.0, 0.0, 0.0, 1.0],
        ]


class CameraTrajectory:
    """Generates continuous smooth camera paths for rendering scene orbits and walkthroughs."""

    @staticmethod
    def orbit(
        target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        radius: float = 3.0,
        elevation_deg: float = 15.0,
        num_frames: int = 60,
        fov_y_deg: float = 45.0,
    ) -> List[Camera]:
        """Generate a complete 360-degree circular orbit trajectory around a target center."""
        cameras = []
        elev_rad = math.radians(elevation_deg)
        y_pos = target[1] + radius * math.sin(elev_rad)
        horiz_radius = radius * math.cos(elev_rad)

        for i in range(num_frames):
            azimuth = 2.0 * math.pi * float(i) / float(num_frames)
            x_pos = target[0] + horiz_radius * math.sin(azimuth)
            z_pos = target[2] + horiz_radius * math.cos(azimuth)

            cam = Camera(
                position=(x_pos, y_pos, z_pos),
                target=target,
                up=(0.0, 1.0, 0.0),
                fov_y_deg=fov_y_deg,
            )
            cameras.append(cam)

        return cameras

    @staticmethod
    def interpolate(cam_a: Camera, cam_b: Camera, t: float) -> Camera:
        """Linearly interpolate between two camera viewpoints."""
        t_clamped = max(0.0, min(1.0, t))
        inv_t = 1.0 - t_clamped

        pos = (
            cam_a.position[0] * inv_t + cam_b.position[0] * t_clamped,
            cam_a.position[1] * inv_t + cam_b.position[1] * t_clamped,
            cam_a.position[2] * inv_t + cam_b.position[2] * t_clamped,
        )
        tgt = (
            cam_a.target[0] * inv_t + cam_b.target[0] * t_clamped,
            cam_a.target[1] * inv_t + cam_b.target[1] * t_clamped,
            cam_a.target[2] * inv_t + cam_b.target[2] * t_clamped,
        )
        fov = cam_a.fov_y_deg * inv_t + cam_b.fov_y_deg * t_clamped

        return Camera(
            position=pos,
            target=tgt,
            up=cam_a.up,
            fov_y_deg=fov,
            near_plane=cam_a.near_plane,
            far_plane=cam_a.far_plane,
        )
