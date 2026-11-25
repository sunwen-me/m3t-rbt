# Read-only gripper pose output

The dashboard exports the latest valid M3T image-associated poses at
`GET /api/overlay/gripper-poses`. This is a preview/capture-rate HTTP interface,
not a 30 Hz control stream. It reads the existing overlay sample, creates no
serial connection, and sends no motor or TF commands.

The response schema is `rbt-m3t-gripper-poses-v1`. `coordinate_frame` is
`orbbec335/color_optical`. Each key in `arms` identifies a distinct physical
SO101 arm (`so101_left` or `so101_right`), with these fields:

| Field | Meaning |
|---|---|
| `T_camera_from_base_link` | M3T's estimated pose of this arm's mechanical base in the camera frame |
| `T_base_link_from_gripper_link` | FK from the runtime model and the exact five joint angles of this M3T estimate |
| `T_camera_from_gripper_link_fk` | Product of the preceding two matrices |
| `T_camera_from_gripper_link_m3t` | Direct `gripper_link` transform from that same native estimate |
| `max_matrix_abs_difference` | Maximum absolute matrix-element difference between composed FK and native output |

All transforms are row-major arrays of 16 numbers, with metre translations and
3×3 rotation matrices. `T_A_from_B` maps column-vector coordinates in B into A.
Translations are at indices 3, 7 and 11. The page displays these values in mm.

```text
T_camera_from_gripper_link_fk
  = T_camera_from_base_link @ T_base_link_from_gripper_link
```

The FK uses radians from the native estimate, converted explicitly to degrees
for the existing model FK function. In the selected encoder-locked mode these
are the applied live encoder measurements, rather than a second current read
that could belong to another image. The native model and M3T output must agree
within `2e-4` per matrix element; this is a software consistency check, not a
physical accuracy specification. Unknown arm/link names, duplicate required
links, invalid rigid transforms and incomplete/nonfinite joint inputs fail.

Frame identity, colour capture timestamp, capture clock identity/generation,
preview age and applied joint-prior diagnostics travel with the response.
The overlay must be enabled, fresh, error-free and associated with a running
M3T experiment. Otherwise this endpoint returns HTTP 503 with an error, never
a successful response containing an old pose. The inherited sample freshness
limit is two seconds; downstream consumers must apply their own task-specific
age and tracking-quality requirements. Received-time feedback does not prove
exposure synchronization.

The same per-arm evidence is stored in `receipt.tcp_poses` in state and saved
verification receipts. The legacy field name does **not** mean a physical
contact TCP has been calibrated: the response explicitly sets
`target_link=gripper_link` and `physical_contact_tcp_calibrated=false`.
No soft-fin contact point, jaw-centre offset or new wrist-camera extrinsic is
invented. A future independently measured `T_gripper_link_from_TCP` can be
composed as a separate step.

## Native TF composition and consumption

Native `perception/M3tTf` output uses the existing `orbbec335/color_optical`
camera frame and an empty default frame prefix. The former `m3t/` prefix is
removed. The canonical paths are:

```text
orbbec335/color_optical → so101_left/cad_mount → so101_left/base_link → … → so101_left/gripper_link
orbbec335/color_optical → so101_right/cad_mount → so101_right/base_link → … → so101_right/gripper_link
```

For each valid capture, `T_C_nominal` is sampled from the active CAD anchor and
`T_C_estimated` is the M3T camera-from-base estimate used above. The nominal
reference is constant for the fixed camera, and the mount correction is
`Δ = inv(T_C_nominal) * T_C_estimated`. Thus
`T_camera_from_base_link = T_C_nominal * Δ`, followed by the same estimate's
encoder-locked URDF chain to `gripper_link`.

The entire native TF path, including the nominal `camera → cad_mount` dynamic
record, carries the same depth capture timestamp. The nominal record is emitted
on every valid capture to make freshness of the complete arm path self-contained.
This differs from the colour capture timestamp carried by the HTTP response.
Native output is scoped to mount refinement with encoder-locked joints and a
fixed camera at an assumed pitch of **−61 degrees**. It does not newly calibrate
pitch, the mount, or a contact TCP. Preserve the robot's `left_base` and
`right_base` legacy references; do not add duplicate canonical static children.

The native query defaults to `orbbec335/color_optical → so101_left/gripper_link`;
select `so101_right/gripper_link` for the right arm. Consumers must subscribe to
`perception/M3tTf` alongside existing robot static and pitch TF as needed for
upstream robot-frame queries. See [native TF output](native-tf-output.md) for
the query command, capture freshness rules and verification boundaries.

## Historical verification and current blocker (2026-11-16)

Local and board Python checks passed 86/86, including nontrivial independent
five-axis FK, both arms, transform direction and units, malformed inputs,
FK/native disagreement, and HTTP 200→503 behavior when data expires. A saved
real bimanual frame produced maximum differences below `1.8e-7` between the
composed and native matrices. That is historical verification, not a live
pose or physical calibration result.

The historical deployment record reports corr=4, Region+Depth,
modeled-occlusion disabled and CAD soft-base/encoder-locked tracking settings.
At deployment the live encoder publisher cache was stale and the interface
returned 503. A later read-only recovery attempt failed on left motor 1's
`Model_Number` read (`comm=-6`), which currently blocks live robot verification.
Fresh output requires restoring encoder feedback and checking new M3T captures.
Prior native software IPC success is historical; it does not establish live
operation or deployment of the canonical CAD-mount correction topology. See
`deploy/20261116-m3t-gripper-output.json` and
`deploy/20261116-m3t-native-tf.json` for historical deployment and rollback details.


The current canonical topology has since passed new native software IPC checks:
three validated captures per arm, 180 complete batches, and stale suppression,
using the same bounded sender as production. The dashboard entrypoint is now the
formal-TF deployment. Fresh real-robot poses remain unavailable for the feedback
and upstream-service reasons above; see `deploy/20261116-m3t-formal-tf.json`.
