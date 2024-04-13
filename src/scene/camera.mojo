"""
Pinhole camera model, view-projection pipelines, and primary ray generation.
"""

from src.core.config import CameraConfig
from src.core.types import CoordinateSystem
from src.math.vector import Vec2f, Vec3f, Vec4f
from src.math.matrix import Mat3x3f, Mat4x4f
from src.math.quaternion import Quatf

struct Ray3f(ImplicitlyCopyable):
    var origin: Vec3f
    var direction: Vec3f

    def __init__(out self, origin: Vec3f, direction: Vec3f):
        self.origin = origin
        self.direction = direction.normalized()

    def at(self, t: Float32) -> Vec3f:
        return self.origin.add(self.direction.mul(t))


struct PinholeCamera(ImplicitlyCopyable):
    """
    Virtual perspective camera with calibrated intrinsics and 6-DOF world pose.
    """
    var config: CameraConfig
    var position: Vec3f
    var rotation: Quatf

    def __init__(out self, config: CameraConfig):
        self.config = config
        self.position = Vec3f(0.0, 0.0, 5.0)
        self.rotation = Quatf.identity()

    def __init__(out self, config: CameraConfig, position: Vec3f, rotation: Quatf):
        self.config = config
        self.position = position
        self.rotation = rotation.normalized()

    @staticmethod
    def look_at(config: CameraConfig, eye: Vec3f, target: Vec3f, up: Vec3f) -> PinholeCamera:
        var f = target.sub(eye).normalized()
        var s = f.cross(up).normalized()
        var u = s.cross(f)

        # Build rotation matrix where columns are s, u, -f
        # Convert to quaternion
        var m00 = s.x
        var m11 = u.y
        var m22 = -f.z
        var trace = m00 + m11 + m22

        var q = Quatf()
        if trace > 0.0:
            from std.math import sqrt
            var r = sqrt(trace + 1.0)
            var inv_s = 0.5 / r
            q.w = 0.5 * r
            q.x = (u.z - (-f.y)) * inv_s
            q.y = ((-f.x) - s.z) * inv_s
            q.z = (s.y - u.x) * inv_s
        else:
            q = Quatf.identity()

        return PinholeCamera(config, eye, q.normalized())

    def get_view_rotation(self) -> Mat3x3f:
        # Inverse rotation of camera pose
        return self.rotation.inverse().to_rotation_matrix()

    def world_to_cam(self, p_world: Vec3f) -> Vec3f:
        var rel = p_world.sub(self.position)
        return self.rotation.inverse().rotate_vec(rel)

    def cam_to_screen(self, p_cam: Vec3f) -> Vec2f:
        var tz = p_cam.z
        if tz < 0.0001 and tz > -0.0001:
            tz = 0.0001
        var u = self.config.focal_x * (p_cam.x / tz) + self.config.principal_x
        var v = self.config.focal_y * (p_cam.y / tz) + self.config.principal_y
        return Vec2f(u, v)

    def world_to_screen(self, p_world: Vec3f) -> Vec2f:
        var p_cam = self.world_to_cam(p_world)
        return self.cam_to_screen(p_cam)

    def generate_ray(self, pixel_x: Float32, pixel_y: Float32) -> Ray3f:
        """
        Cast primary camera ray from optical center through specified image pixel.
        """
        var x_norm = (pixel_x - self.config.principal_x) / self.config.focal_x
        var y_norm = (pixel_y - self.config.principal_y) / self.config.focal_y
        var dir_cam = Vec3f(x_norm, y_norm, 1.0).normalized()
        var dir_world = self.rotation.rotate_vec(dir_cam)
        return Ray3f(self.position, dir_world)
