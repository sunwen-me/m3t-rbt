# SO101 calibration tools

## 夹爪贴码网页（左臂）

`aruco_web.py` 复用现有主相机预览包及只读 v2 反馈；使用
`DICT_4X4_50 / ID 1`、黑色外框实测 29 mm。启动参数与
`shared_board_web.py` 的主相机、模型、驱动和输出参数一致，不需要夹爪相机。
只支持当前固定的左臂配置，输出必须使用独立目录，不混入双相机样本。

采集时固定主相机、底座和刚性贴码位置，只移动机械臂；移走同号的
ChArUco 板。网页拒绝同号重复、小于 25 px 的码、关节移动及过期帧，
记录原始图像、四次只读反馈和来源哈希。软件接收时间不是曝光同步。
至少六个姿态可计算，建议 10–15 个；沿用训练、运动多样性、满秩和
留一姿态验证门槛，结果不会自动写入 M3T 初始化。

求解器支持单臂或双臂，只对样本实际包含的臂输出结果。初始化使用
当前拟合子集的相对运动估计夹爪到码的变换，留出样本不参与初始化。

### 固定云台的模型叠加验证

`aruco_web.py` 可加 `--overlay-audit FILE --overlay-report FILE
--overlay-static-config FILE`，用已部署安装边的审计及对应候选报告启动 `/verify`。
`--overlay-pitch-status-url` 指定既有云台控制器的只读状态地址。
点击开始后，后台读取 v2 反馈，等待新的主相机帧，再读取反馈；移动、
失联、超出模型限位或云台偏离假定采集角度超过 0.5° 时不显示叠加。
超过两秒的处理结果不再显示。时间配对是主机时间，并非曝光同步。

此页面不启用 M3T、PnP 或其他视觉结果修正模型姿态。网格按原始反馈 FK
及固定外参投影，预测码框与实测码框仅用于比较。活动夹指及其软鳍暂时
省略，不能把夹爪百分比直接当作 URDF 弧度。轮廓包含被其他部件遮挡的
边，属于透视轮廓显示。保存验证画面时同时保存原图、叠加图和反馈来源，
不增加或改写标定样本。

## Shared ChArUco board website

`wrist_intrinsics.py` collects the wrist camera's own intrinsics;
`shared_board_web.py` combines its preview with the existing main-camera RGB
packet and read-only v2 motor feedback. `shared_board.py` solves the saved
observations offline. These tools do not command motor motion or automatically
replace M3T initialization. Camera images and calibration sessions belong in
an ignored output directory or the robot's calibration directory, outside Git.

The confirmed board has **5 columns × 8 rows of squares**, 24 mm square sides,
17 mm markers, `DICT_4X4_50`, and `legacy_pattern: false`. Do not substitute
the nominal 25/18 mm dimensions from its original SVG filename.

1. Keep the wrist camera still, place the board fully in its view, and lock
   the current focus. Save at least 12 different board positions and tilts,
   then compute the wrist intrinsics. Move the board during this phase.
2. Adopt a passing intrinsic candidate, then fix the board, main camera,
   pitch mount and arm base. Manually move the arm into different stationary
   poses with both cameras seeing the board. Save each pose through the site.
3. Solve after at least six distinct poses with rotation about different axes.
   Training, held-out reprojection and observability checks must pass. A
   passing report measures consistency; independent physical verification
   remains necessary before runtime deployment.

The website uses software receipt timing, not hardware exposure synchronization.
Only the left wrist camera is currently connected; a left-arm report makes no
claim about the right arm. Each wrist camera needs its own intrinsic calibration.

Offline dependencies and focused tests:

```bash
python -m pip install -r requirements-shared-board.txt
python -m unittest discover -s . -p 'test_shared_board.py'
python -m unittest discover -s . -p 'test_wrist_intrinsics.py'
python -m unittest discover -s . -p 'test_shared_board_web.py'
```

The servers bind to loopback. Forward the dashboard port over SSH to operate
them from the workstation. See each script's `--help` for explicit source and
output arguments; do not open an already-owned camera or arm serial port.

## ArUco mount calibration from a fixed external camera

`handeye.py` fits the per-arm fixed transforms

```
T_camera_from_marker(i) = T_camera_from_base
                        @ T_base_from_gripper(q_i)
                        @ T_gripper_from_marker
```

It minimizes reprojection of the four decoded ArUco corners. The camera, arm
bases and marker attachments must stay fixed during the dataset; only joints
move. Camera intrinsics are fixed and images must already be rectified. Encoder
FK supplies robot motion; M3T tracking results do not enter this solver.

The manifest contains `schema_version: 1`, a confirmed `setup_id`,
`setup_confirmed_unchanged: true`, `marker_side_m`, per-arm nominal
`T_gripper_from_marker` transforms, and `samples`. Each sample has an `id`,
`pose_group`, and observations under `arms.so101_left` / `arms.so101_right`:
`T_base_from_gripper`, `corners_px`, `K`, and `T_camera_from_marker_seed`.
Transforms are 4x4 matrices in metres with column-vector convention. Corners
are decoded canonical TL, TR, BR, BL, corresponding to marker coordinates
`(-s/2,+s/2), (+s/2,+s/2), (+s/2,-s/2), (-s/2,-s/2)`.

An observation can also supply `depth_plane` with `normal_camera` (unit vector),
`offset_m`, and `sigma_m`. These describe a plane measured independently from
registered depth: `normal_camera · point_camera + offset_m = 0`. The solver
adds four plane-distance residuals with one combined plane weight. Never derive
the plane from the fitted pose or the PnP seed. Reports distinguish RGB and
RGB-D inputs and include depth residuals in millimetres.

Requires NumPy and SciPy. It never communicates with the robot or writes motor
goals. Output is exclusively created, so prior results cannot be overwritten.

```bash
python handeye.py --manifest samples.json --output result.json
python -m unittest discover -s . -p 'test_handeye.py'
```

Acceptance requires at least four materially distinct poses per arm,
nonparallel rotational motion, a full-rank scaled Jacobian, positive camera
depths, and bounded training and leave-one-pose-group-out reprojection errors.
Repeated frames do not count as independent poses. Failed gates produce
candidate transforms with reasons; they do not authorize replacing runtime TF.
For RGB-D input the training plane RMS must also be at most 5 mm, every training
frame at most 8 mm, and each heldout frame at most 10 mm.

Even accepted results measure cross-pose consistency, not independently known
absolute accuracy. They depend on encoder calibration, URDF kinematics, camera
intrinsics and marker size. The wrist camera's optical extrinsics are a separate
calibration problem. Review fit stability and physical mount plausibility before
using a result to generate URDF or runtime initialization.

The unified SO101 driver owns both serial ports and publishes v2 encoder states
for the web dashboard and native tracker. Pass `--m3t-prior-service PREFIX/state/v1`
and `--joint-feedback-controller` to `aruco_web.py`. The dashboard uses
`--driver-python` to subscribe through the installed RBT SDK and checks the
controller identity, the complete dual-arm sample and its freshness.
Five arm angles become radian soft priors with
standard deviation 0.1 rad. The gripper percentage is separately mapped through
the actual v2 encoder endpoint travel (1435 ticks = 126.1538° for current left arm) with
standard deviation 0.01 rad. Both initialization and continuous gripper priors
use that physical travel. The closed endpoint is assigned nominal model zero;
its alignment to the URDF zero remains unverified.
All six actual applied diagnostics must pass, with maximum received-time age
100 ms, before corrected overlays are shown. This is software timing, not
exposure synchronization. The experiment fixes the base and requires pitch
near −61°. Continuous priors do not independently validate camera/base calibration.

The verification renderer now displays both gripper jaws and both soft fins.
For the cyan feedback overlay, the sixth feedback value is an opening percentage:
0% is the actual calibrated closed endpoint; angular displacement follows the
v2 raw endpoint span and installed driver resolution (4095), without scaling
to URDF limits. The closed endpoint is provisionally assigned model angle zero;
this zero alignment is not independently measured. It does not alter encoder calibration, motor
limits. A session-local tracking model widens the gripper limit to accommodate
the measured travel without modifying the canonical URDF. Orange M3T geometry uses all actual
per-link output transforms, including the moving jaw and attached fin.

The verification page has independent visibility controls for the cyan encoder
model, orange M3T result (including its predicted marker), pink encoder-predicted
marker, and green observed marker. These controls affect display only; tracking,
feedback freshness checks and error calculations continue. Saved receipts record
the selected display layers, alongside the unchanged tracking diagnostics.


## 固定 −61° 的双臂名义初值入口（2026-11-15）

`aruco_web.py --dual-arms --right-calibration FILE --right-arm-port /dev/so101_right`
在原左臂参数之外增加右臂来源；必须提供 `--nominal-mount-prior`、
`--overlay-static-config` 和 `--m3t-prior-service`。该服务由统一 SO101 驱动按
左右 v2 标定发布。`--fixed-pitch-assumption-deg -61` 显式使用固定角度假设。

双臂共用一个 `M3TExperiment`、一份含两臂的 initialization、一个 native Tracker
和同一帧 RGB-D；每臂独立 Link 树、Optimizer、固定 base 与六项关节软先验。
双臂 packet 和 12 项 `(arm,joint)` 身份必须完整，禁止拼接两个独立跟踪进程的图像。
页面橙色为左臂、紫色为右臂，青色编码器叠加可选。模型副本的限位调整须单独留存来源。
板端本次入口为 `/home/aidlux/rbt-work/tools/shared-board/start-bimanual-dashboard.sh`；
部署与回退见 `deploy/20261115-m3t-bimanual-init.json`。

## v2 与 URDF 的关节零点（2026-11-15）

用户确认当前 v2 标定零点就是当前 URDF 零点。前五轴继续使用
`q_model_rad = v2_degree * pi / 180`，两臂均不另加零位偏置。
已捕获的实际驱动、双臂初始化及 12 项原生先验数值链一致。
右肘的记录行程越界与模型限位越界分别报告，不通过裁剪或偏置掩盖。
[角度核查及可重复命令](../docs/so101-v2-model-angle-audit.md)说明输入、依据及验收范围。

正式模型的 `elbow_flex` 上限现为 100°，已覆盖右肘 v2 反馈 97.758°。
模型 hash 与固定 −61° prior 的来源绑定同步更新；
[板端限位更新记录](../../../deploy/20261115-so101-elbow-upper-limit.json)记录当前入口及回退路径。

## CAD 安装修正实验

`--m3t-mount-refinement` 在固定相机、nominal mount prior 与完整实时反馈下，
将所有关节固定为编码器值，仅对每臂 base 使用围绕 CAD 的六自由度软约束。
`--m3t-end-effector-priority` 是独立的夹爪刚性结构 5 倍视觉权重选项。
当前默认选用前者；后者在本轮画面中未通过匹配检查，保留为实验选项。
[配置、数值处理与当前边界](../docs/cad-mount-refinement.md)。

## Read-only gripper poses

`GET /api/overlay/gripper-poses` exports both arms' camera→base, base→gripper
and composed camera→gripper matrices from one verified overlay estimate.
The page shows per-arm XYZ in mm; saved receipts include `tcp_poses`.
The target is `gripper_link`, **not a calibrated physical contact TCP**.
Stale or unavailable tracking returns HTTP 503.
See [the pose contract and current verification limits](../docs/gripper-pose-output.md).
