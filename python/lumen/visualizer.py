"""
Visualization, ASCII terminal preview, and standard image export tools.
"""

from typing import List


def export_ppm(filepath: str, width: int, height: int, rgb_bytes: bytes) -> None:
    """Save an RGB buffer to a binary Portable Pixmap (PPM P6) image file."""
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(rgb_bytes)


def export_depth_visualization(
    filepath: str,
    width: int,
    height: int,
    depth_values: List[float],
    near_plane: float = 0.5,
    far_plane: float = 10.0,
) -> None:
    """Save a depth buffer to a colorized heatmap PPM image."""
    pixel_count = width * height
    rgb_buf = bytearray(pixel_count * 3)

    for i in range(pixel_count):
        d = depth_values[i]
        if d >= 999.0 or d <= 0.0:
            norm_d = 1.0
        else:
            norm_d = (d - near_plane) / (far_plane - near_plane)
            norm_d = max(0.0, min(1.0, norm_d))

        # Invert so close objects are bright and far objects are dark
        intensity = int((1.0 - norm_d) * 255.0)

        # Apply false-color gradient (blue to green to yellow)
        r = int(intensity * 1.0)
        g = int(intensity * 0.8)
        b = int(255 - intensity * 0.7)

        rgb_buf[i * 3 + 0] = max(0, min(255, r))
        rgb_buf[i * 3 + 1] = max(0, min(255, g))
        rgb_buf[i * 3 + 2] = max(0, min(255, b))

    export_ppm(filepath, width, height, bytes(rgb_buf))


def ascii_render(
    width: int,
    height: int,
    alpha_values: List[float],
    target_width: int = 60,
    target_height: int = 25,
) -> str:
    """Render an ASCII density block preview of the spatial scene."""
    chars = " .:-=+*#%@"
    num_levels = len(chars) - 1

    scale_x = width / target_width
    scale_y = height / target_height

    lines = []
    for y in range(target_height):
        orig_y = int(y * scale_y)
        if orig_y >= height:
            orig_y = height - 1
        line_chars = []
        for x in range(target_width):
            orig_x = int(x * scale_x)
            if orig_x >= width:
                orig_x = width - 1

            idx = orig_y * width + orig_x
            alpha = max(0.0, min(1.0, alpha_values[idx]))
            char_idx = int(alpha * num_levels)
            line_chars.append(chars[char_idx])
        lines.append("".join(line_chars))

    return "\n".join(lines)
