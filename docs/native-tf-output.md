# M3T native RBT TF output

`rbt-m3t run --config tracking.json --publish-tf` publishes native RBT dynamic
transforms directly in the tracking component activation. The advertised
capability is `perception/M3tTf`. No HTTP polling, dashboard sampling or motor
connection is involved in this publication path; valid estimates still require
fresh encoder feedback from the existing feedback source.

The official topology uses the existing camera frame and canonical arm frames.
The default frame prefix is empty; the former `m3t/` prefix is removed. The
robot's existing `left_base` and `right_base` static children remain legacy
references. Do not add duplicate static children for the canonical arm frames.

```text
orbbec335/color_optical
├── so101_left/cad_mount                  (nominal CAD reference)
│   └── so101_left/base_link             (tracking base reference)
│       └── so101_left/shoulder_link → … → so101_left/gripper_link
└── so101_right/cad_mount                 (nominal CAD reference)
    └── so101_right/base_link            (tracking base reference)
        └── so101_right/shoulder_link → … → so101_right/gripper_link
```

The publisher supports three modes with a fixed camera. CAD mount refinement
alone estimates the base with encoder-locked joints. Encoder reference links
alone use fixed tracking bases and visually refined joints. Enabling both
`encoder_reference_links` and `mount_refinement.enabled` refines the base under
the CAD mount soft constraint while retaining visual joint optimization under
encoder soft priors. Its output reports
`mount_refinement.joints_locked_to_feedback=false`; mount refinement being
enabled does not imply locked joints. The camera pitch assumption is
**−61 degrees**.

For each arm, let `T_C_nominal` be the nominal camera-from-base transform sampled
from the CAD anchor, and `T_C_estimated` be the tracking base in the same camera
frame. In either encoder reference mode, `tf_nominal_initialization` can supply
the CAD anchors separately from the tracking seeds in `initialization`. Its
camera and arm identities must match. The optimizer's nominal mount diagnostic
is the initialization prior center and may differ from this TF CAD anchor.
The TF anchor remains constant while the current estimated base determines
the correction on every capture. The composition is:

```text
T_camera_from_cad_mount = T_C_nominal
Δ = inv(T_C_nominal) * T_C_estimated
T_cad_mount_from_base_link = Δ
T_camera_from_base_link = T_C_nominal * Δ = T_C_estimated
```

The remaining URDF parent-child edges through `gripper_link` use the same
frame's estimated poses. Each link edge is computed as
`inverse(T_camera_parent) * T_camera_child`. Translation is in metres and
rotation uses the native RBT quaternion XYZW convention. `gripper_link` is not
an independently calibrated contact TCP; no contact-point frame is invented.
Combined mode validates every mount diagnostic's arm identity, rigid nominal
and refined poses, correction magnitudes and agreement with the frame's root.
It uses the fused visual joint poses without replacing them with encoder FK.
The independent dashboard subscriber displays the packet's actual base and
joint mode, along with per-arm mount corrections.

The entire arm path carries the RGB-D pair's **depth capture timestamp** and
clock identity, matching the existing M3T pose timestamp contract. Even the
constant nominal `camera → cad_mount` reference is published as a dynamic record
on every valid capture, so a fresh complete path is self-contained in M3T TF.
Capture uncertainty is preserved, including unknown uncertainty. A missing,
incompatible, future or more than 500 ms old timestamp produces no TF records.
Inference completion time is used only to test freshness, never to restamp old
poses. The required joint-feedback gate prevents estimates from being computed
from unavailable or stale feedback.

`transform_epoch` stays zero during a publisher activation. Frame sequence is
kept in the existing tracking JSON and is **not** stored in the epoch field.
A new activation receives a new capability authority from the RBT runtime.
Native TF transport publishes one edge per message, not an atomic tree batch;
consumers must query an available complete path at one capture timestamp.
The SDK Control stream has four unread slots per subscriber. The shared native
sender retries the same record at 500 microsecond intervals when a queue is full,
with a 25 ms wait budget per batch and a fresh age check before every attempt.
Stop, stale timing or sustained pressure abandons the remaining batch. Retries
retain the exact record, timestamp and epoch; they do not refresh old poses.
Queue pressure is reported. Individual consumers still require a complete path
at a single capture, because transport is not atomic.

The independent AidLux `rbt-m3t-bimanual.service` owns the tracking process and
TF publication. The dashboard reads the native preview packet selected by
`--m3t-preview-packet`. Its process can restart independently. The dual-arm
publisher keeps the existing RBT component ID for clients that select it.

## Consumption

The native query tool resolves the capability through Composer and checks the
provider identity. The default query is from `orbbec335/color_optical` to
`so101_left/gripper_link`. Example for the independent native component:

```bash
rbt-m3t-tf-query \
  --provider m3t-bimanual-web-experiment \
  --parent orbbec335/color_optical \
  --child so101_left/gripper_link \
  --count 3 --timeout-sec 10
```

Use `so101_right/gripper_link` for the right arm. The canonical left path is
`orbbec335/color_optical → so101_left/cad_mount → so101_left/base_link → … → so101_left/gripper_link`;
the right path substitutes `so101_right` throughout.

Application consumers must subscribe to `perception/M3tTf` and ingest its
dynamic records into their `rbt::tf::TransformBuffer`, alongside existing robot
static and pitch TF subscriptions as needed for queries from upstream robot
frames. Advertising a capability does not automatically subscribe an existing
listener. The query tool accepts unknown timing uncertainty explicitly and
requires a fresh complete path, with bounded timeout when data is unavailable.

## Verification boundaries

CPU conversion checks cover software behavior such as transform composition,
wire encode/decode, native Buffer lookup, identity/shape validation, capture
clock preservation and stale suppression. An isolated native software fixture
uses `fixture/color_optical` and `m3t_fixture/...`, never the real camera/tree.
Software checks do not establish physical calibration or accuracy. Real-robot
validation requires fresh encoder feedback and a running M3T estimate stream.

## Initial software verification and deployment (2026-11-16)

The canonical topology passed three native test programs locally and on AidLux,
and 91 Python checks on each host. Two independent native consumers queried
three distinct captures per arm through Composer. Translation, quaternion,
nominal/correction/model edges, authority and epoch were checked. The shared
production sender delivered 180 complete fixture batches (1080 edge records),
recovered from 14 queue-full events and dropped no batches. After updates stopped,
a new query timed out with exit code 2 and emitted no pose. The initial unpaced
fixture failed under the four-slot queue limit; the shared bounded sender above
is the correction verified by these results.

The dashboard and persistent launcher now select canonical TF output, retaining
corr=4, Region+Depth, CAD soft bases and encoder-locked joints. Live encoder
feedback remains stale after the earlier motor-1 Model_Number read failure
(`comm=-6`). Robot static and pitch providers were also absent during the live
read-only audit, and the pitch HTTP status server was unavailable. Therefore no
fresh robot TF or complete base_footprint-to-gripper path is claimed. The actual
robot-provider query timed out without emitting a pose. No upstream service,
calibration, encoder offset or motor state was changed in this task.

See `deploy/20261116-m3t-formal-tf.json` for the original binary, software
results and rollback. The previous `deploy/20261116-m3t-native-tf.json` describes
the historical isolated `m3t/` topology.
