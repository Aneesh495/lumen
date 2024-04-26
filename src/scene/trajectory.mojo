"""
Continuous camera trajectories, orbit paths, and Catmull-Rom spline flight generation.
"""

from .camera import PinholeCamera
from src.core.config import CameraConfig
from src.core.types import TWO_PI_F32, PI_F32
from src.math.vector import Vec3f
from src.math.quaternion import Quatf
from std.collections import List
from std.math import cos, sin

struct CameraKeyframe(ImplicitlyCopyable):
    var time: Float32
    var position: Vec3f
    var rotation: Quatf

    def __init__(out self, time: Float32, position: Vec3f, rotation: Quatf):
        self.time = time
        self.position = position
        self.rotation = rotation.normalized()


def catmull_rom_spline(p0: Vec3f, p1: Vec3f, p2: Vec3f, p3: Vec3f, t: Float32) -> Vec3f:
    """
    Evaluate Catmull-Rom cubic spline interpolation at local parameter t in [0, 1].
    """
    var t2 = t * t
    var t3 = t2 * t

    # Tangents: m1 = 0.5 * (p2 - p0), m2 = 0.5 * (p3 - p1)
    # Hermite blending functions
    var f1 = -0.5 * t3 + t2 - 0.5 * t
    var f2 = 1.5 * t3 - 2.5 * t2 + 1.0
    var f3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    var f4 = 0.5 * t3 - 0.5 * t2

    return Vec3f(
        p0.x * f1 + p1.x * f2 + p2.x * f3 + p3.x * f4,
        p0.y * f1 + p1.y * f2 + p2.y * f3 + p3.y * f4,
        p0.z * f1 + p1.z * f2 + p2.z * f3 + p3.z * f4
    )


struct CameraTrajectory:
    """
    Manages smooth camera flight paths and automated novel-view evaluation trajectories.
    """
    var keyframes: List[CameraKeyframe]
    var base_config: CameraConfig

    def __init__(out self, config: CameraConfig):
        self.keyframes = List[CameraKeyframe]()
        self.base_config = config

    def add_keyframe(mut self, frame: CameraKeyframe):
        self.keyframes.append(frame)

    def keyframe_count(self) -> Int:
        return len(self.keyframes)

    @staticmethod
    def generate_orbit(
        config: CameraConfig,
        center: Vec3f,
        radius: Float32,
        height: Float32,
        num_frames: Int = 60
    ) -> CameraTrajectory:
        """
        Generate a circular orbit trajectory centered at specified target point.
        """
        var traj = CameraTrajectory(config)
        var step_angle = TWO_PI_F32 / Float32(num_frames)

        for i in range(num_frames):
            var theta = Float32(i) * step_angle
            var eye = Vec3f(
                center.x + radius * cos(theta),
                center.y + height,
                center.z + radius * sin(theta)
            )
            var up = Vec3f(0.0, 1.0, 0.0)
            var cam = PinholeCamera.look_at(config, eye, center, up)
            var t_norm = Float32(i) / Float32(num_frames)
            traj.add_keyframe(CameraKeyframe(t_norm, cam.position, cam.rotation))

        return traj^

    def sample(self, t_norm: Float32) -> PinholeCamera:
        """
        Sample interpolated camera pose along trajectory at parameter in [0, 1].
        """
        var n = len(self.keyframes)
        if n == 0:
            return PinholeCamera(self.base_config)
        if n == 1:
            return PinholeCamera(self.base_config, self.keyframes[0].position, self.keyframes[0].rotation)

        var t_clamped = t_norm
        if t_clamped < 0.0: t_clamped = 0.0
        elif t_clamped > 1.0: t_clamped = 1.0

        var segment_float = t_clamped * Float32(n - 1)
        var idx = Int(segment_float)
        if idx >= n - 1:
            idx = n - 2
        var local_t = segment_float - Float32(idx)

        # 4 control points for Catmull-Rom
        var idx0 = idx - 1
        if idx0 < 0: idx0 = 0
        var idx1 = idx
        var idx2 = idx + 1
        var idx3 = idx + 2
        if idx3 >= n: idx3 = n - 1

        var pos = catmull_rom_spline(
            self.keyframes[idx0].position,
            self.keyframes[idx1].position,
            self.keyframes[idx2].position,
            self.keyframes[idx3].position,
            local_t
        )

        var rot = self.keyframes[idx1].rotation.slerp(self.keyframes[idx2].rotation, local_t)
        return PinholeCamera(self.base_config, pos, rot)
