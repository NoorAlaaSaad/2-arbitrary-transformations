"""Transform KITTI lidar points into an imaginary camera and render three views.

Run ``python arbitrary_transformations.py --help`` for examples of the inputs.
Angles are in degrees, coordinates are in metres, and pixels use (column, row).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 720
DEFAULT_DATA_DIRECTORY = Path(__file__).resolve().parent / "data" / "raw"


def load_lidar_points(path):
    """Read KITTI's binary records: float32 x, y, z, and reflectance."""
    path = Path(path)
    if path.stat().st_size == 0 or path.stat().st_size % 16 != 0:
        raise ValueError(f"{path} must contain complete, nonempty KITTI records.")

    records = np.fromfile(path, dtype="<f4").reshape(-1, 4)
    # Reflectance is not a coordinate or a homogeneous-coordinate value.
    points = records[:, :3].astype(np.float64)
    if not np.isfinite(points).all():
        raise ValueError(f"{path} contains nonfinite coordinates.")
    return points


def rotation_matrix(roll_degrees, pitch_degrees, yaw_degrees):
    """Return Rz(yaw) @ Ry(pitch) @ Rx(roll) for column-vector coordinates."""
    roll, pitch, yaw = np.deg2rad([roll_degrees, pitch_degrees, yaw_degrees])
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

    rotation_x = np.array([
        [1, 0, 0],
        [0, cos_roll, -sin_roll],
        [0, sin_roll, cos_roll],
    ])
    rotation_y = np.array([
        [cos_pitch, 0, sin_pitch],
        [0, 1, 0],
        [-sin_pitch, 0, cos_pitch],
    ])
    rotation_z = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1],
    ])
    return rotation_z @ rotation_y @ rotation_x


def rigid_transform(
    roll_degrees=0.0,
    pitch_degrees=0.0,
    yaw_degrees=0.0,
    translation=(0.0, 0.0, 0.0),
):
    """Build a 4x4 transform for p_destination = R @ p_source + translation."""
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix(roll_degrees, pitch_degrees, yaw_degrees)
    transform[:3, 3] = translation
    return transform


def transform_points(points, transform):
    """Apply a homogeneous transform to an array with one 3D point per row."""
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    # Transpose R because the array stores row vectors instead of column vectors.
    return points @ rotation.T + translation


def camera_intrinsics(
    fx=800.0,
    fy=800.0,
    principal_x=IMAGE_WIDTH / 2,
    principal_y=IMAGE_HEIGHT / 2,
):
    """Build K with a centered principal point in (column, row) coordinates.

    The handout swaps the dimensions; use (360, 960) to reproduce it literally.
    """
    return np.array([
        [fx, 0, principal_x],
        [0, fy, principal_y],
        [0, 0, 1],
    ], dtype=float)


def project_points(camera_points, intrinsics, width, height):
    """Return visible integer (u, v) pixels and their forward camera depths."""
    # Remove points behind or too close to the camera before dividing by depth.
    valid_depth = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 0.1)
    visible_points = camera_points[valid_depth]
    depths = visible_points[:, 2]
    homogeneous_pixels = visible_points @ intrinsics.T
    pixels = homogeneous_pixels[:, :2] / depths[:, None]

    inside_image = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    # Each integer pixel covers [u, u+1) x [v, v+1).
    return np.floor(pixels[inside_image]).astype(int), depths[inside_image]


def render_projection(
    camera_points,
    intrinsics,
    width=IMAGE_WIDTH,
    height=IMAGE_HEIGHT,
    max_depth=80.0,
):
    """Paint nearest visible points on black using a 256-color HSV depth scale."""
    if max_depth <= 0 or not np.isfinite(max_depth):
        raise ValueError("max_depth must be a finite positive distance.")

    image = np.zeros((height, width, 3), dtype=np.uint8)
    pixels, depths = project_points(camera_points, intrinsics, width, height)
    if len(pixels) == 0:
        return image

    # The handout's 'hsv ' has a trailing space; the colormap name is 'hsv'.
    color_map = plt.get_cmap("hsv", 256)
    color_table = (color_map(np.arange(256))[:, :3] * 255).astype(np.uint8)
    color_indices = (np.clip(depths / max_depth, 0, 1) * 255).astype(int)

    # Sort by depth, then keep the first (nearest) point at each pixel.
    nearest_first = np.argsort(depths, kind="stable")
    pixel_ids = pixels[:, 1] * width + pixels[:, 0]
    _, first_occurrences = np.unique(pixel_ids[nearest_first], return_index=True)
    selected = nearest_first[first_occurrences]
    columns, rows = pixels[selected].T
    image[rows, columns] = color_table[color_indices[selected]]
    return image


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scan", nargs="?", type=Path,
        help="KITTI .bin scan; defaults to the first scan under data/raw/",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"),
        help="directory for PNG images (default: outputs)",
    )
    parser.add_argument(
        "--azimuths", type=float, nargs="+", default=[0.0, 45.0, 90.0],
        help="viewer azimuths in degrees, positive toward lidar left",
    )
    parser.add_argument(
        "--translation", type=float, nargs=3, default=[0.0, 0.0, 0.0],
        metavar=("TX", "TY", "TZ"),
        help="lidar-to-camera translation in camera axes, in metres",
    )
    parser.add_argument(
        "--principal-point", type=float, nargs=2,
        default=[IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2], metavar=("CX", "CY"),
        help="default: centered 960 360; use 360 960 for the literal handout values",
    )
    parser.add_argument(
        "--max-depth", type=float, default=80.0,
        help="HSV color scale maximum in metres; farther points keep the last color",
    )
    parser.add_argument(
        "--show", action="store_true", help="also open the comparison figure",
    )
    args = parser.parse_args()
    if not np.isfinite(args.azimuths + args.translation + args.principal_point).all():
        parser.error("angles, translation, and principal point must be finite")
    if not np.isfinite(args.max_depth) or args.max_depth <= 0:
        parser.error("--max-depth must be a finite positive distance")
    if args.scan is None:
        scans = sorted(DEFAULT_DATA_DIRECTORY.glob("**/velodyne_points/data/*.bin"))
        if not scans:
            parser.error("no KITTI scans found; provide a .bin file or extract Scene 5 into data/raw/")
        args.scan = scans[0]
    if not args.scan.is_file():
        parser.error(f"scan does not exist: {args.scan}")
    return args


def main():
    args = parse_arguments()
    if not args.show:
        plt.switch_backend("Agg")
    lidar_points = load_lidar_points(args.scan)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Lidar: x forward, y left, z up. Camera: x right, y down, z forward.
    # (roll, pitch, yaw) = (90, -90, 0) gives (x_c, y_c, z_c) = (-y, -z, x).
    # The imaginary camera shares the lidar origin unless translation is supplied.
    lidar_to_camera = rigid_transform(90, -90, 0, args.translation)
    intrinsics = camera_intrinsics(principal_x=args.principal_point[0],
                                   principal_y=args.principal_point[1])
    np.set_printoptions(precision=4, suppress=True)
    print(f"Loaded {len(lidar_points):,} points from {args.scan}")
    print("Euler angles [roll, pitch, yaw] (degrees): [90, -90, 0]")
    print(f"Translation [x, y, z] (metres): {args.translation}")
    print(f"Lidar-to-camera transform:\n{lidar_to_camera}")
    print(f"Camera intrinsics K:\n{intrinsics}")

    figure, axes = plt.subplots(
        len(args.azimuths), 1, figsize=(12, 4.8 * len(args.azimuths)), squeeze=False,
    )
    for axis, azimuth in zip(axes[:, 0], args.azimuths):
        # Camera Y is vertical. Ry(+azimuth) expresses points in a viewer
        # looking left by azimuth degrees, with the same camera center.
        camera_to_viewer = rigid_transform(pitch_degrees=azimuth)
        lidar_to_viewer = camera_to_viewer @ lidar_to_camera
        viewer_points = transform_points(lidar_points, lidar_to_viewer)

        # The full 3x4 projection is P = K @ lidar_to_viewer[:3, :].
        # Project in viewer coordinates so points behind the viewer can be removed.
        projection_matrix = intrinsics @ lidar_to_viewer[:3, :]
        print(f"Projection matrix at {azimuth:g} degrees:\n{projection_matrix}")
        image = render_projection(viewer_points, intrinsics, max_depth=args.max_depth)
        output_path = args.output_dir / f"azimuth_{azimuth:g}.png"
        plt.imsave(output_path, image)
        occupied_pixels = np.count_nonzero(np.any(image != 0, axis=2))
        print(f"Saved {output_path} ({occupied_pixels:,} occupied pixels)")
        axis.imshow(image, interpolation="nearest")
        axis.set_title(f"Viewer azimuth: {azimuth:g} degrees")
        axis.set_axis_off()

    figure.suptitle(
        f"KITTI Scene 5 | HSV: camera depth 0–{args.max_depth:g} m (clipped)\n"
        f"Principal point: ({args.principal_point[0]:g}, {args.principal_point[1]:g})"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    comparison_path = args.output_dir / "comparison.png"
    figure.savefig(comparison_path, dpi=120)
    print(f"Saved {comparison_path}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
