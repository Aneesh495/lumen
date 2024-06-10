"""
Top-level SIMD tile rasterizer orchestrating culling, sorting, binning, and pixel compositing.
"""

from src.core.config import RenderConfig
from src.core.buffer import FloatBuffer, UInt64Buffer, IntBuffer
from src.scene.gaussian_cloud import GaussianCloud
from src.scene.camera import PinholeCamera
from src.math.vector import Vec2f, Vec3f
from src.math.projection import Covariance2D, compute_ewa_covariance
from src.math.cholesky import solve_eigenvalues_2x2
from src.math.spherical_harmonics import evaluate_sh_color
from .tile import TileGrid
from .culling import FrustumCuller
from .radix_sort import encode_tile_depth_key, radix_sort_64bit
from .binning import TileRangeIndex
from .compositor import PixelAccumulator, compute_gaussian_pixel_alpha
from .depth_map import DepthMap
from std.collections import List

struct RenderOutput:
    var width: Int
    var height: Int
    var rgb_buffer: FloatBuffer
    var depth_buffer: FloatBuffer
    var visible_splats: Int
    var total_splat_instances: Int

    def __init__(out self, width: Int, height: Int):
        self.width = width
        self.height = height
        self.rgb_buffer = FloatBuffer(width * height * 3, 0.0)
        self.depth_buffer = FloatBuffer(width * height, 0.0)
        self.visible_splats = 0
        self.total_splat_instances = 0

    def get_pixel_rgb(self, x: Int, y: Int) -> Vec3f:
        if x >= 0 and x < self.width and y >= 0 and y < self.height:
            var idx = (y * self.width + x) * 3
            return Vec3f(
                self.rgb_buffer.get(idx + 0),
                self.rgb_buffer.get(idx + 1),
                self.rgb_buffer.get(idx + 2)
            )
        return Vec3f(0.0, 0.0, 0.0)


def rasterize_scene(
    cloud: GaussianCloud,
    camera: PinholeCamera,
    config: RenderConfig
) -> RenderOutput:
    """
    Execute full tile-based SIMD rasterization pass over 3D Gaussian radiance field.
    """
    var width = camera.config.width
    var height = camera.config.height
    var output = RenderOutput(width, height)

    var grid = TileGrid(width, height, config.tile_width, config.tile_height)
    var culler = FrustumCuller(camera)

    var n_gaussians = cloud.count()
    if n_gaussians == 0:
        return output^

    var view_rot = camera.get_view_rotation()

    # Pre-allocate projected properties for surviving Gaussians
    var centers_x = List[Float32](capacity=n_gaussians)
    var centers_y = List[Float32](capacity=n_gaussians)
    var depths = List[Float32](capacity=n_gaussians)
    var inv_covs = List[Covariance2D](capacity=n_gaussians)
    var colors = List[Vec3f](capacity=n_gaussians)
    var opacities = List[Float32](capacity=n_gaussians)

    # 64-bit instance keys and values
    var instance_keys = UInt64Buffer(n_gaussians * 4)
    var instance_vals = IntBuffer(n_gaussians * 4)

    var visible_count = 0

    # 1. Geometry and Culling Pass
    for i in range(n_gaussians):
        var g = cloud.get_gaussian(i)
        if not culler.is_opacity_visible(g.opacity):
            continue

        var p_cam = camera.world_to_cam(g.position)
        if not culler.is_depth_visible(p_cam.z):
            continue

        # Projected 2D screen center
        var screen_pt = camera.cam_to_screen(p_cam)

        # 3D and 2D Covariance
        var cov3d = g.compute_covariance()
        var cov2d = compute_ewa_covariance(
            cov3d,
            p_cam,
            view_rot,
            camera.config.focal_x,
            camera.config.focal_y,
            config.lowpass_filter
        )

        var eigen = solve_eigenvalues_2x2(cov2d)
        if eigen.radius < 1:
            continue

        if not culler.is_screen_box_visible(screen_pt.x, screen_pt.y, eigen.radius):
            continue

        var min_tx = 0
        var min_ty = 0
        var max_tx = 0
        var max_ty = 0

        var overlaps = grid.get_gaussian_tile_overlap(
            screen_pt.x,
            screen_pt.y,
            eigen.radius,
            min_tx,
            min_ty,
            max_tx,
            max_ty
        )

        if not overlaps:
            continue

        # Evaluate viewing direction and color
        var view_dir = g.position.sub(camera.position).normalized()
        var splat_color = evaluate_sh_color(
            cloud.soa.sh_data,
            i * 48,
            view_dir,
            config.max_sh_degree
        )

        var curr_visible_idx = visible_count
        visible_count += 1

        centers_x.append(screen_pt.x)
        centers_y.append(screen_pt.y)
        depths.append(p_cam.z)
        inv_covs.append(cov2d.inverse())
        colors.append(splat_color)
        opacities.append(g.opacity)

        # Emit tile keys
        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                var tile_id = grid.get_tile_id(tx, ty)
                var key = encode_tile_depth_key(UInt32(tile_id), p_cam.z)
                instance_keys.append(key)
                instance_vals.append(curr_visible_idx)

    output.visible_splats = visible_count
    output.total_splat_instances = instance_keys.size()

    if instance_keys.size() == 0:
        return output^

    # 2. 64-bit Radix Sort
    radix_sort_64bit(instance_keys, instance_vals)

    # 3. Tile Range Construction
    var tile_index = TileRangeIndex.build_from_sorted_keys(instance_keys, grid.total_tiles)

    # 4. Per-Tile SIMD Pixel Compositing
    for ty in range(grid.tiles_y):
        for tx in range(grid.tiles_x):
            var tile_id = grid.get_tile_id(tx, ty)
            var range_info = tile_index.get_range(tile_id)

            var px_start = grid.get_tile_pixel_min_x(tx)
            var px_end = grid.get_tile_pixel_max_x(tx)
            var py_start = grid.get_tile_pixel_min_y(ty)
            var py_end = grid.get_tile_pixel_max_y(ty)

            if range_info.is_empty():
                # Background fill
                if config.white_background:
                    for py in range(py_start, py_end):
                        for px in range(px_start, px_end):
                            var p_idx = (py * width + px) * 3
                            output.rgb_buffer.set(p_idx + 0, 1.0)
                            output.rgb_buffer.set(p_idx + 1, 1.0)
                            output.rgb_buffer.set(p_idx + 2, 1.0)
                continue

            # Composite each pixel in tile
            for py in range(py_start, py_end):
                var py_center = Float32(py) + 0.5
                for px in range(px_start, px_end):
                    var px_center = Float32(px) + 0.5
                    var acc = PixelAccumulator()

                    for inst_idx in range(range_info.start, range_info.end):
                        var vis_id = instance_vals.get(inst_idx)

                        var alpha = compute_gaussian_pixel_alpha(
                            px_center,
                            py_center,
                            centers_x[vis_id],
                            centers_y[vis_id],
                            inv_covs[vis_id],
                            opacities[vis_id]
                        )

                        var terminated = acc.blend(
                            colors[vis_id],
                            alpha,
                            depths[vis_id]
                        )

                        if terminated:
                            break

                    acc.finalize(config.white_background)

                    var out_idx = (py * width + px) * 3
                    output.rgb_buffer.set(out_idx + 0, acc.color.x)
                    output.rgb_buffer.set(out_idx + 1, acc.color.y)
                    output.rgb_buffer.set(out_idx + 2, acc.color.z)

                    if config.compute_depth:
                        output.depth_buffer.set(py * width + px, acc.depth)

    return output^
