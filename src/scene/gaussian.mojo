"""
3D Gaussian primitives with Array of Structures (AoS) and Structure of Arrays (SoA) layouts.
"""

from src.math.vector import Vec3f
from src.math.quaternion import Quatf
from src.math.covariance import Covariance3D
from src.core.types import MAX_SH_COEFFS_PER_GAUSSIAN
from std.collections import List
from std.math import exp

struct Gaussian3D(ImplicitlyCopyable):
    """
    Individual 3D Gaussian representation in Array of Structures (AoS) format.
    """
    var position: Vec3f
    var scale: Vec3f
    var rotation: Quatf
    var opacity: Float32

    def __init__(out self):
        self.position = Vec3f(0.0, 0.0, 0.0)
        self.scale = Vec3f(0.1, 0.1, 0.1)
        self.rotation = Quatf.identity()
        self.opacity = 1.0

    def __init__(
        out self,
        position: Vec3f,
        scale: Vec3f,
        rotation: Quatf,
        opacity: Float32
    ):
        self.position = position
        self.scale = scale
        self.rotation = rotation.normalized()
        self.opacity = opacity

    def compute_covariance(self) -> Covariance3D:
        return Covariance3D.from_scale_and_rotation(self.scale, self.rotation)

    @staticmethod
    def opacity_from_logit(logit: Float32) -> Float32:
        return 1.0 / (1.0 + exp(-logit))


struct GaussianSoA:
    """
    Cache-optimized Structure of Arrays (SoA) storage for multi-million Gaussian collections.
    """
    var pos_x: List[Float32]
    var pos_y: List[Float32]
    var pos_z: List[Float32]

    var scale_x: List[Float32]
    var scale_y: List[Float32]
    var scale_z: List[Float32]

    var rot_w: List[Float32]
    var rot_x: List[Float32]
    var rot_y: List[Float32]
    var rot_z: List[Float32]

    var opacity: List[Float32]
    var sh_data: List[Float32]

    def __init__(out self, capacity: Int = 0):
        self.pos_x = List[Float32](capacity=capacity)
        self.pos_y = List[Float32](capacity=capacity)
        self.pos_z = List[Float32](capacity=capacity)

        self.scale_x = List[Float32](capacity=capacity)
        self.scale_y = List[Float32](capacity=capacity)
        self.scale_z = List[Float32](capacity=capacity)

        self.rot_w = List[Float32](capacity=capacity)
        self.rot_x = List[Float32](capacity=capacity)
        self.rot_y = List[Float32](capacity=capacity)
        self.rot_z = List[Float32](capacity=capacity)

        self.opacity = List[Float32](capacity=capacity)
        self.sh_data = List[Float32](capacity=capacity * MAX_SH_COEFFS_PER_GAUSSIAN)

    def count(self) -> Int:
        return len(self.pos_x)

    def append(mut self, g: Gaussian3D, sh: List[Float32]):
        self.pos_x.append(g.position.x)
        self.pos_y.append(g.position.y)
        self.pos_z.append(g.position.z)

        self.scale_x.append(g.scale.x)
        self.scale_y.append(g.scale.y)
        self.scale_z.append(g.scale.z)

        self.rot_w.append(g.rotation.w)
        self.rot_x.append(g.rotation.x)
        self.rot_y.append(g.rotation.y)
        self.rot_z.append(g.rotation.z)

        self.opacity.append(g.opacity)

        for i in range(MAX_SH_COEFFS_PER_GAUSSIAN):
            if i < len(sh):
                self.sh_data.append(sh[i])
            else:
                self.sh_data.append(0.0)

    def get_gaussian(self, index: Int) -> Gaussian3D:
        var pos = Vec3f(self.pos_x[index], self.pos_y[index], self.pos_z[index])
        var scale = Vec3f(self.scale_x[index], self.scale_y[index], self.scale_z[index])
        var rot = Quatf(self.rot_w[index], self.rot_x[index], self.rot_y[index], self.rot_z[index])
        var op = self.opacity[index]
        return Gaussian3D(pos, scale, rot, op)
