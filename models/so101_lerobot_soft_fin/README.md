# SO-101 LeRobot soft-fin visual model

User-confirmed M3T model baseline (2026-11-12): the user confirmed this model
after viewing it in URDF Visualizer. Future SO-101 M3T model generation,
initialization, configuration, and tracking work must use
`so101_lerobot_soft_fin.urdf` from this recipe unless the user explicitly
requests a model change. Do not revert to the legacy numbered-joint model.
This confirmation accepts the displayed geometry and model selection; it does
not establish physical calibration or deployment acceptance.

This is the canonical model recipe for M3T's SO-101 follower with the four-part
soft-finger geometry and a fixed #34 bracket accessory. It starts from the
official follower URDF `so101/so101_new_calib.urdf` in the Hugging Face
`lerobot/robot-urdfs` dataset at revision
`04d113b3a280b5d27264dbbf56f9a98c94b68321`. The exact source and every vendored
official asset are recorded with SHA-256 hashes in `upstream.lock.json`; the
official files and custom geometry are local, so rebuilding does not fetch
assets from the network.

The official active joints keep their names, origins, and axes:
`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, and
`gripper`. The generated model expands `shoulder_lift` and `wrist_flex` to
±110° and the `elbow_flex` upper limit from 1.69 rad to 100°
(`math.radians(100)`), retaining its lower limit of -1.69 rad. At the user's
request, `wrist_roll` covers [-π, +π] rad (±180°) to match both arms' v2 raw
calibration range [0, 4095], normalized by `(raw - midpoint) * 360 / 4095`.
This includes the recorded right wrist angle of -168.66°. All other official
joint limits and semantics remain pinned to the source. These project-owned
overrides leave the upstream URDF and `upstream.lock.json` unchanged. They cover
v2 recorded travel up to 105.45° for `shoulder_lift` and 102.77° for `wrist_flex`,
and the recorded right elbow angle of 97.758°, with a small margin. The builder
records each old and generated range in `model_provenance.json` and rejects
unapproved joint changes. Added fixed geometry frames are mapped using common
CAD anchors. The mapping does not transfer the old model's arm zero or limits.
The official `gripper_frame_link` fixed joint is preserved as the nominal
official TCP; it is not a calibration of the soft-fin contact point.

The model reuses the previous assembly's fixed short palm, moving jaw body,
fixed fin, and moving fin. The #34 bracket is attached as a fixed accessory. The
old `camera_module` and `handeye_cam` branch is omitted; this model defines no
camera pose or hand-eye calibration. Do not carry over the old model's
initialization, base poses, encoder offsets, or numeric-joint wrist mappings:
regenerate initialization from a current RGB-D snapshot and measured joint
angles, and review any driver-to-model offsets against the actual robot.

The upstream SO-101 source is attributed under Apache-2.0; the corresponding
license text is in [`licenses/LICENSE`](licenses/LICENSE). That attribution
does not assert a license for the custom soft-fin or #34 bracket geometry.
Inherited inertial values are not recalculated for the added geometry. Fins are
rigid in this model, so TPU deformation, contact behavior, dynamics, and
collision safety are not validated. This is a visual-tracking model, not a
hardware acceptance or deployment record.

## Build the model and M3T robot files

Run these commands from the repository root. Generated files, snapshots, and
initialization stay under `target/`:

```bash
export M3T_DATA="$PWD/target/so101-lerobot-soft-fin"
mkdir -p "$M3T_DATA"
python3 perception/m3t/models/so101_lerobot_soft_fin/build_model.py \
  --output "$M3T_DATA/model"
python3 perception/m3t/scripts/prepare_robot.py \
  --urdf "$M3T_DATA/model/so101_lerobot_soft_fin.urdf" \
  --output-dir "$M3T_DATA/robot"
```

The builder writes the curated model directory by default when `--output` is
omitted. Use `--output DIR` to place a generated copy outside the source tree.
`prepare_robot.py` requires its output directory not to exist yet.

The model build does not create an M3T initialization. Start Composer and the
aligned camera, then collect a fresh snapshot:

```bash
# Terminal 1
composer

# Terminal 2
rbt-orbbec-camera run --serial "$CAMERA_SERIAL" --align-depth-to-color

# Terminal 3
rbt-m3t probe --provider orbbec335 --count 30 --duration-sec 10 \
  --snapshot "$M3T_DATA/snapshot" --preview
```

Record the actual arm joint angles at the snapshot time in radians in
`$M3T_DATA/joint_angles.json`, then run the initializer. The output file must
not already exist:

```bash
rbt-m3t-initialize \
  --model "$M3T_DATA/robot/robot.json" \
  --snapshot "$M3T_DATA/snapshot" \
  --output "$M3T_DATA/initialization.json" \
  --joint-angles "$M3T_DATA/joint_angles.json"
```

The configuration example
[`../../configs/so101_lerobot_soft_fin.example.json`](../../configs/so101_lerobot_soft_fin.example.json)
uses paths that resolve to this `target/` layout when used directly from the
repository. If you copy it elsewhere, update `robot_model`, `initialization`,
and `cache_directory` relative to the copied config. The initialization path
becomes valid only after the measured initialization step above. Run with:

```bash
rbt-m3t run --config perception/m3t/configs/so101_lerobot_soft_fin.example.json
```

## Tests and validation status

Run the model-builder regression with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s perception/m3t/models/so101_lerobot_soft_fin \
  -p 'test_build_model.py' -v
```

The recorded software and file checks are:

- `test_build_model.py`: 9/9 passed after the four limit overrides, including
  the unchanged elbow lower limit, the recorded 97.758° elbow angle, and wrist
  coverage of -180°, -168.66°, and +180°.
- Existing `prepare_robot.py` regression suite: 6/6 passed in 3.054 s.
- Independent SHA-256 and size checks passed for the pinned upstream URDF and
  all 13 vendored official assets.
- The generated URDF converted to 11 links and 10 joints, including the six
  official revolute joints. Its 20 visual meshes merged to 10 nonempty OBJ
  links; the nominal official TCP has no mesh.
- URDFHub loaded the generated filename and named joints without missing
  meshes. The browser preview is at
  `target/so101-lerobot-soft-fin/web-preview/preview.jpg`. Its flattened mesh
  URIs are preview-only; the canonical generated URDF was unchanged.

These checks cover model generation, conversion, and browser preview only. No
real camera or arm commands were issued, and no new initialization,
calibration, physical installation, or deployment was validated.
