# Arbitrary transformations

This assignment solution loads a KITTI Scene 5 lidar scan, transforms it into an
imaginary camera coordinate system, adds viewer azimuths of 0°, 45°, and 90°, and
projects each view onto a black 1920 × 720 image. Point colors show forward camera
depth using a 256-color HSV colormap.

The submission file is `arbitrary_transformations.py`. It contains the complete
solution; `tests/` checks the geometry and `requirements.txt` lists its dependencies.

## Run the solution

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python arbitrary_transformations.py
```

The script automatically finds the first scan under
`data/raw/**/velodyne_points/data/`. The supplied Scene 5 dataset is already there.
For a fresh checkout, download [KITTI Scene 5](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip)
and extract it into `data/raw/`, or pass a scan path directly:

```bash
python arbitrary_transformations.py /path/to/0000000000.bin
```

The default run saves `azimuth_0.png`, `azimuth_45.png`, `azimuth_90.png`, and
`comparison.png` in `outputs/`. It also prints the Euler angles, translation,
rigid transform, camera intrinsics, and each complete 3 × 4 projection matrix.
Add `--show` to open the comparison in a graphical session. Saved images work
without a display.

The default principal point is centered at `(960, 360)`. The handout appears to
swap its dimensions; see the explanation below. To reproduce its literal values
and keep those results separately:

```bash
python arbitrary_transformations.py --principal-point 360 960 --output-dir outputs/literal
```

Other parameters are optional:

```bash
python arbitrary_transformations.py --azimuths 0 45 90 --translation 0 0.2 0 --max-depth 60
python arbitrary_transformations.py --help
```

`--translation` is the extrinsic translation expressed in the destination camera
axes, in metres. It is **not** the camera's position in lidar coordinates.
The depth color scale defaults to 0–80 metres. Points farther than 80 metres
remain visible but use the last color. Each view uses the same scale.

## 1. Lidar-to-camera transformation

The assignment diagrams use these axes:

| Axis | Lidar | Camera |
| --- | --- | --- |
| x | forward | right |
| y | left | down |
| z | up | forward |

Therefore the required coordinate mapping is:

```text
x_camera = -y_lidar
y_camera = -z_lidar
z_camera =  x_lidar
```

For column vectors, the Euler convention is `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`;
the rightmost rotation acts first. One valid set of Euler angles is
**[roll, pitch, yaw] = [90°, -90°, 0°]**:

```text
    [ 0  -1   0 ]
R = [ 0   0  -1 ]
    [ 1   0   0 ]

    [ R  t ]
T = [ 0  1 ]       p_camera = R @ p_lidar + t
```

The imaginary camera is assumed to share the lidar origin, giving
**t = [0, 0, 0] metres**. The assignment supplies no physical camera offset to
estimate, so this is an explicit modeling choice. If the camera center is `C`
in lidar coordinates, use `t = -R @ C`. For example, a camera 0.2 m above the
lidar has `C = [0, 0, 0.2]` and `t = [0, 0.2, 0]`.

Euler angles are not unique here: pitch −90° is a gimbal-lock configuration.
The resulting rotation matrix is still valid and has determinant +1.

The code stores points as rows, so it evaluates `points @ R.T + t`, which is
equivalent to the column-vector equation above. KITTI's fourth binary value is
reflectance; it is discarded, not used as a homogeneous coordinate.

## 2. Additional viewer transformation

The viewer shares the imaginary camera's center. Positive azimuth turns its
viewing direction toward the lidar's left side. Because camera Y is vertical,
the coordinate transformation into this viewer is a rotation around camera Y:

```text
T_camera_to_viewer = rigid_transform(pitch_degrees=azimuth)
T_lidar_to_viewer  = T_camera_to_viewer @ T_lidar_to_camera
```

Thus the 0°, 45°, and 90° viewers look along lidar directions `(1, 0, 0)`,
`(1/√2, 1/√2, 0)`, and `(0, 1, 0)`, respectively. Applying the viewer transform
on the left ensures it acts after the lidar-to-camera transform.

## 3. Projection and rendering

The focal lengths are `fx = fy = 800` pixels. With pixels indexed as
`image[row, column]`, the pinhole equations are:

```text
    [ 800    0   cx ]
K = [   0  800   cy ]
    [   0    0    1 ]

u = 800 * X / Z + cx       # column
v = 800 * Y / Z + cy       # row
P = K @ T_lidar_to_viewer[:3, :]   # complete 3x4 projection matrix
```

The handout explicitly says `PPX = Height/2` and `PPY = Width/2`, giving
**cx = 360, cy = 960** if interpreted literally. With conventional column/row
coordinates, this puts the principal point below the 720-pixel image. On the
first supplied scan, all three projections are completely black with these
values. This appears to be a dimension swap in the handout.

The solution therefore defaults to a centered camera: **cx = Width/2 = 960**
and **cy = Height/2 = 360**. This is an explicit interpretation of the ambiguous
handout, not a change to the requested focal lengths or image size. The
`--principal-point 360 960` option reproduces the literal values. Neither choice
changes the coordinate transforms.

Before division, points with nonfinite coordinates or depth `Z <= 0.1 m` are
removed. Only projections within the image are drawn. Pixel coordinates are
floored; when points share a pixel, the nearest depth wins. HSV colors use
`plt.get_cmap("hsv", 256)`, correcting the handout's trailing space in `"hsv "`.

## Verify

```bash
python -m unittest discover -s tests -v
```

The tests cover binary loading, axis mapping, rigid distances, composition
order, all three viewer directions, projection bounds, hidden and empty point
clouds, and visibility at overlapping pixels. Datasets, generated images,
virtual environments, caches, and local configuration are excluded from Git.
