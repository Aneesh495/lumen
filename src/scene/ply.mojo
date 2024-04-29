"""
Stanford PLY point cloud reader and writer for 3D Gaussian Splatting scene data.
"""

from .gaussian import Gaussian3D
from .gaussian_cloud import GaussianCloud
from src.math.vector import Vec3f
from src.math.quaternion import Quatf
from src.core.error import LumenStatus, StatusCode
from src.core.types import MAX_SH_COEFFS_PER_GAUSSIAN, parse_float
from std.collections import List

struct PLYHeader(ImplicitlyCopyable):
    var vertex_count: Int
    var is_binary: Bool
    var has_normals: Bool
    var sh_degree: Int

    def __init__(out self):
        self.vertex_count = 0
        self.is_binary = False
        self.has_normals = False
        self.sh_degree = 0

    def __init__(out self, count: Int, is_binary: Bool = False):
        self.vertex_count = count
        self.is_binary = is_binary
        self.has_normals = False
        self.sh_degree = 3


def write_ply_ascii(cloud: GaussianCloud, filepath: String) -> LumenStatus:
    """
    Serialize Gaussian cloud into standard Stanford PLY ASCII format.
    """
    var count = cloud.count()
    try:
        with open(filepath, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("comment Created by Lumen 3D Gaussian Splatting Engine\n")
            f.write("element vertex " + String(count) + "\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float f_dc_0\n")
            f.write("property float f_dc_1\n")
            f.write("property float f_dc_2\n")
            f.write("property float opacity\n")
            f.write("property float scale_0\n")
            f.write("property float scale_1\n")
            f.write("property float scale_2\n")
            f.write("property float rot_0\n")
            f.write("property float rot_1\n")
            f.write("property float rot_2\n")
            f.write("property float rot_3\n")
            f.write("end_header\n")

            for i in range(count):
                var g = cloud.get_gaussian(i)
                var sh0 = cloud.soa.sh_data[i * MAX_SH_COEFFS_PER_GAUSSIAN + 0]
                var sh1 = cloud.soa.sh_data[i * MAX_SH_COEFFS_PER_GAUSSIAN + 1]
                var sh2 = cloud.soa.sh_data[i * MAX_SH_COEFFS_PER_GAUSSIAN + 2]

                var line = String(g.position.x) + " " + String(g.position.y) + " " + String(g.position.z) + " "
                line += String(sh0) + " " + String(sh1) + " " + String(sh2) + " "
                line += String(g.opacity) + " "
                line += String(g.scale.x) + " " + String(g.scale.y) + " " + String(g.scale.z) + " "
                line += String(g.rotation.w) + " " + String(g.rotation.x) + " " + String(g.rotation.y) + " " + String(g.rotation.z) + "\n"
                f.write(line)

        return LumenStatus.ok()
    except e:
        return LumenStatus.error(StatusCode.IO_ERROR, "Failed to write PLY file: " + String(e))


def read_ply_ascii(filepath: String) -> GaussianCloud:
    """
    Parse an ASCII Stanford PLY file into a GaussianCloud.
    """
    var cloud = GaussianCloud()
    try:
        with open(filepath, "r") as f:
            var content = f.read()
            var lines = content.split("\n")

            var in_header = True

            for i in range(len(lines)):
                var line = lines[i].strip()
                if line.byte_length() == 0:
                    continue

                if in_header:
                    if line == "end_header":
                        in_header = False
                else:
                    var tokens = line.split(" ")
                    if len(tokens) >= 14:
                        var px = parse_float(String(tokens[0]))
                        var py = parse_float(String(tokens[1]))
                        var pz = parse_float(String(tokens[2]))

                        var sh0 = parse_float(String(tokens[3]))
                        var sh1 = parse_float(String(tokens[4]))
                        var sh2 = parse_float(String(tokens[5]))

                        var op = parse_float(String(tokens[6]))

                        var sx = parse_float(String(tokens[7]))
                        var sy = parse_float(String(tokens[8]))
                        var sz = parse_float(String(tokens[9]))

                        var rw = parse_float(String(tokens[10]))
                        var rx = parse_float(String(tokens[11]))
                        var ry = parse_float(String(tokens[12]))
                        var rz = parse_float(String(tokens[13]))

                        var g = Gaussian3D(
                            Vec3f(px, py, pz),
                            Vec3f(sx, sy, sz),
                            Quatf(rw, rx, ry, rz),
                            op
                        )

                        var sh = List[Float32](capacity=48)
                        sh.append(sh0)
                        sh.append(sh1)
                        sh.append(sh2)
                        for _ in range(45):
                            sh.append(0.0)

                        cloud.add_gaussian(g, sh)

        return cloud^
    except e:
        print("PLY read warning:", e)
        return cloud^
