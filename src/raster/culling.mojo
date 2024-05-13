"""
View frustum culling, camera near-plane clipping, and screen boundary validation.
"""

from src.scene.camera import PinholeCamera
from src.math.vector import Vec3f
from src.core.types import MIN_ALPHA_THRESHOLD

struct CullingStatistics(ImplicitlyCopyable):
    var total_gaussians: Int
    var culled_behind_camera: Int
    var culled_near_far: Int
    var culled_off_screen: Int
    var culled_transparent: Int
    var surviving: Int

    def __init__(out self):
        self.total_gaussians = 0
        self.culled_behind_camera = 0
        self.culled_near_far = 0
        self.culled_off_screen = 0
        self.culled_transparent = 0
        self.surviving = 0


struct FrustumCuller(ImplicitlyCopyable):
    """
    Evaluates whether 3D and 2D projected Gaussians intersect the camera visible volume.
    """
    var near_plane: Float32
    var far_plane: Float32
    var screen_width: Int
    var screen_height: Int

    def __init__(out self, camera: PinholeCamera):
        self.near_plane = camera.config.near_plane
        self.far_plane = camera.config.far_plane
        self.screen_width = camera.config.width
        self.screen_height = camera.config.height

    def is_depth_visible(self, cam_z: Float32) -> Bool:
        """
        Check if point is strictly in front of near plane and within far plane.
        """
        return cam_z >= self.near_plane and cam_z <= self.far_plane

    def is_opacity_visible(self, opacity: Float32) -> Bool:
        """
        Check if Gaussian has sufficient alpha to contribute to rendered pixel color.
        """
        return opacity >= MIN_ALPHA_THRESHOLD

    def is_screen_box_visible(
        self,
        center_x: Float32,
        center_y: Float32,
        radius: Int
    ) -> Bool:
        """
        Test whether the 3-sigma bounding circle of the 2D projected Gaussian intersects the screen.
        """
        var r_f = Float32(radius)
        if center_x + r_f < 0.0:
            return False
        if center_x - r_f >= Float32(self.screen_width):
            return False
        if center_y + r_f < 0.0:
            return False
        if center_y - r_f >= Float32(self.screen_height):
            return False
        return True
