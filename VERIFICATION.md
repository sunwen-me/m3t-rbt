# 验证记录

## 2026-11-13：左腕 UVC 重枚举恢复与过期画面处理

现场确认同一物理 USB 路径的 Sunplus `1bcf:2281` 从 video2 重枚举到 video3，
旧句柄最后一帧年龄超过 700 秒。设备没有独立硬件序列号；udev 的
`XHH-260128-A_2M` 来自厂商/产品字符串。新身份使用厂商、产品与固定物理端口
绑定，`/dev/videoN` 仅作为采集时的历史字段，不再决定内参身份。

按原预检的 USB 路径及设备属性核对并恢复 AF=0、focus=0 后，对 18 个元数据
JSON 的相机身份标签做了审计迁移；K、D、角点和原始 PNG 未变，采用的候选哈希
同步更新。原件与审计在 aidlux `runs/wrist-usb-rebind-*`，12 组内参数据保留。

采集线程现会在读帧失败时清除缓存，按同一身份及模式退避重连，并在重新验证
锁焦状态后提供画面；无法确认身份、分辨率或焦距时拒绝使用。网页清除过期图像，
掉线时暂停采集。腕部测试 11 项（含模拟重连/身份与焦距拒绝）、网页测试 14 项
通过；aidlux 腕部测试同样 11 项通过。现场重新采集后 `connected=true`、
`error=null`，画面年龄约 55 ms，AF/focus 仍为 0。自动重连分支通过模拟测试，
本次未人为再次拔插 USB。另已恢复工作站 SSH 转发，并加入连接保活。

## 2026-11-13：按 v2 行程扩展两处模型限位

用户确认来源 URDF 限位过窄后，将 `shoulder_lift` 从约 ±100°、`wrist_flex`
从约 ±95°，均扩展到 ±110°。两臂 v2 文件记录的对应行程最大约为
105.45° / 102.77°。覆盖仅位于项目派生模型；上游固定版本与原始来源文件未改。
生成器明确记录这两处 override，并继续严格检查其他官方关节语义。

新 URDF SHA-256 为 `ded58ad7d5f4aee710cdce91c37015caf6bf28d51c784b5153720880316b872a`。
本机与 aidlux 已重生成 robot JSON，逐关节比对确认除这四个 lower/upper 字段外
关节定义相同，OBJ 网格字节相同。网页模型 pin 和既有手眼工作阶段的模型 hash
同步更新；迁移审计保存在该阶段 `model-limit-migration-*.json`，保留内参采用记录。
当时尚未保存手眼姿态（0 组），12 组内参采集及候选结果保留。

模型构建测试 9 项、URDF 转换测试 6 项、网页测试 11 项通过；网页实际只读
检查显示“v2 寄存器与五关节反馈检查通过”。未向电机写入目标、扭矩或硬件限位。
同时发现夹爪 UVC 摄像头重新枚举到 video3，原采集进程读帧失败，需恢复相机连接
后再继续手眼采集；本记录不表示手眼标定已完成。

## 2026-11-13：共享 ChArUco 板的本地操作网站

同日后续：用户完成左腕相机 12 组真实采集，保存的内参候选整体 RMS
`0.48907367146731673 px`，质量门槛通过。原同步计算超过代理 10 秒等待时间，
返回成功结果时触发 `BrokenPipeError`，被错误写成采集状态错误；数据与结果
文件均未丢失。代理现已捕获客户端断线且不重放 POST，超时返回 503；
新增代理断线/超时回归后该测试文件共 11 项通过。腕部计算改为单个后台任务，
通过状态轮询显示进度和结果，并按设备、焦距、分辨率和板参数恢复已存候选。
修复后的腕部采集器 7 项测试通过；重启实测恢复原 12 组数据和同一候选，
网页再次点击计算立即返回后台任务提示，计算期间预览/状态继续更新，完成后
`error: null`，新保存候选 RMS 仍为 `0.48907367146731673 px`。
此处的内参候选通过不代表共享板手眼标定或 M3T 初始化已通过。

新增 `calibration/wrist_intrinsics.py`、`shared_board_web.py` 和离线
`shared_board.py`。站点运行在 aidlux 的 loopback 8097/8098，通过工作站
SSH 转发访问；复用现有主相机预览包，不重新占用 Orbbec USB。
左臂新接入的 UVC 相机使用固定物理 USB 路径、1920×1080 MJPG。
板参数由用户确认：5×8 方格、24 mm 格边、17 mm 码边、4X4_50、非 legacy。

本机独立环境通过求解器 9 项、网站辅助逻辑 9 项和内参采集器 5 项测试。
aidlux 使用系统 OpenCV 4.13.0、NumPy 1.26.4 和独立环境 SciPy 1.15.3，
求解器同样通过 9 项离线测试。浏览器验证了两路真实预览、锁定自动对焦、
保存第一组 28 角点的原始 PNG/观测、重复姿态拒绝，以及服务重启后保留
1/12 内参姿态和锁焦状态。未生成真实内参或手眼结果，采集数据不入 Git。

当前只接左臂夹爪相机；右臂未验收。网站通过独立子进程只读 v2 电机寄存器和
反馈，不调用目标写入或扭矩控制。最新实测左臂串口存在，但 motor 1 的
`Model_Number` 读取 `comm=-6`，页面会提示检查供电/USB。机械臂反馈恢复、
内参采集和多姿态手眼采集完成前，不能声称真实手眼标定通过；候选也不会自动
写入 M3T 初始化。

## 2026-11-12：旧 ArUco 候选标定的 URDF 坐标转换

对比冻结旧 URDF 与用户确认的官方 soft-fin 模型，四个共同底座 CAD 网格给出的
`T_old_base_from_new_base` 最大矩阵元素差小于 `1e-7`。以共同 base 网格为锚点，
该变换近似绕 Z 轴 `-90.00021°`，平移为旧 base 坐标下的
`[0.020790886, 0.015760790, 0.032481700] m`。
迁移公式为 `T_camera_from_new_base = T_camera_from_old_base @ T_old_base_from_new_base`。
腕部的等价关节参数为 `q_new_wrist_roll = q_old_joint5 - 87.21108056°`；其他活动关节对应同名机械关节。

独立组合 100 组关节姿态，以新旧底座转换后的 shoulder 至 jaw 位姿比较，最大位置差
`2.80e-6 m`、最大旋转差 `0.000670°`，与源 URDF 数值舍入量级一致。
旧标定实际使用的 robot JSON 六个活动关节 origin 也与冻结旧 URDF 一致。
转换脚本、报告和候选初始化位于本机 `target/so101-lerobot-soft-fin/legacy-frame-migration/`；
候选和报告已放到板端本次只读 run 目录，尚未接入活动配置。

这证明旧外参可以进行坐标转换，并不消除旧拟合的残余误差。
候选保留旧采样相机姿态及旧拟合关节零点约定，不可把其中的关节角当成当前 v2 反馈。
是否可用于当前图像仍取决于相机、云台、两臂基座的相对安装状态；重新安装的码不使用旧码安装变换。

## 2026-11-12：真实相机与 v2 双从臂只读反馈

实时检查 `ssh aidlux`：Gemini335 `CP15641000AT` 的 M3T probe 收到 10 组对齐 RGB-D；USB 重新接入后，左右 CH343 串口均识别，12 个 STS3215 只读返回反馈。两臂全部舵机的 Homing_Offset、Min_Position_Limit、Max_Position_Limit 与电脑上的 follower v2 校准一致。按用户要求，板端两个驱动入口的校准文件改为 v2，旧内容被移除；电脑缓存中旧 follower 文件也已删除。未修改舵机寄存器、使能或运动目标，读取时 12 个 Torque_Enable 均为 0。

证据在板端 `runs/m3t-so101-lerobot-live-20261112-x_cjm3o1/`：`servo-calibration-registers.json`、`v2-calibration-selection.json`、`joint-readonly-v2.json`。`snapshot-with-joints/` 与 `snapshot-joint-association.json` 保存同一采集窗口的真实 RGB-D 和关节原始值；关联是同主机单调接收时间的 nearest read，不是曝光同步，且此前按旧文件计算的 normalized 值不可作为 v2 初始化。

启动测试修正了专用配置中的绝对路径，模型、初始化和缓存现使用相对配置文件的路径。修正后跟踪启动明确停在新模型 `initialization.json` 缺失；未生成伪造初始化，也未建立真实跟踪成功结论。现有双相机采集进程保持运行。舵机 v2 校准不等于 M3T 相机到基座的初始化。

## 2026-11-12：LeRobot SO-101 soft-fin 模型生成与转换

本轮验证 Hugging Face `lerobot/robot-urdfs` revision `04d113b3a280b5d27264dbbf56f9a98c94b68321` 的官方 follower URDF、13 个 vendored 官方 mesh 的 SHA-256/文件大小、定制 soft-fin 与 #34 bracket 几何，以及生成 URDF 的 M3T 转换和浏览器预览。证据范围仅限软件与文件；没有操作真实相机或机械臂，也没有生成新 initialization、相机标定或部署验收记录。

| 项目 | 结果与证据 |
| --- | --- |
| 模型构建回归 | `test_build_model.py`：8/8 PASS，3.871 s。 |
| `prepare_robot.py` 回归 | 6/6 PASS，3.054 s。 |
| 官方源文件完整性 | 官方来源 URDF 与 13 个官方资产的 SHA-256 和文件大小独立复核 PASS。 |
| URDF 转换 | `target/so101-lerobot-soft-fin/robot/robot.json`：11 links、10 joints、6 个官方 revolute joints；20 个 visual mesh 合并为 10 个非空 OBJ link；官方 nominal TCP 无 mesh。 |
| 浏览器预览 | URDFHub 加载新 URDF 文件名和命名关节，无 missing mesh；截图为 `target/so101-lerobot-soft-fin/web-preview/preview.jpg`。浏览器仅在预览副本中展平 mesh URI，规范生成 URDF 未修改。 |

模型来源、构建和初始化步骤见 [LeRobot soft-fin 模型说明](models/so101_lerobot_soft_fin/README.md)。这里的通过结果不代表实机安装、相机/hand-eye 标定、软鳍接触点精度或部署已验证。

同日通过 `ssh aidlux` 将用户确认的模型文件同步到板端
`/home/aidlux/rbt-work/models/so101-lerobot-soft-fin/`。实时读取板端文件独立复核：
`model/` 的 32 个文件与本机 SHA-256 完全一致；在板端从最终路径转换出的
`robot/robot.json` 含 11 links、10 joints、6 个官方命名活动关节和 10 个非空 OBJ。
`source_urdf` 及全部 mesh 来源路径在板端存在，来源哈希一致。本轮只同步模型资料和
配置，不启动跟踪、不修改现有运行配置；新模型的实测 initialization 仍未生成。

## 2026-11-11：原生双路 RGB-D 联合跟踪

源码新增可选 `secondary_camera` 配置，每视角创建相机、模态和渲染器，复用物体模型并向同一 Link/每臂 optimizer 累加观测。第二路 Orbbec 通过 `--secondary-camera` 发布独立的 Secondary 能力类型，避免当前 SDK 将两个同类型 requirement 解析到同一提供者。每路独立 D2C/标定与 RGB-D 配对后，再有界配对两路时间，不复用旧帧。默认单路配置保持兼容。

以下均为本机 Release 构建及合成软件验证，未操作物理相机或机械臂，未新增板端部署或实机验收结论。

| 项目 | 结果与证据 |
| --- | --- |
| M3T 模块测试 | 9/9 PASS；`target/m3t-dual-camera-build/Testing/Temporary/LastTest.log`。包含跨视角时间配对、时钟身份/replay/队列边界、严格配置与输出、双视角输入校验及主相机坐标输出，并回归原有关节先验。 |
| Orbbec 驱动测试 | 7/7 PASS；`target/m3t-dual-camera-driver-build/Testing/Temporary/LastTest.log`。包含主/次角色配置，以及软件后端的 Secondary Stream、CameraInfo/Control RPC、静态 TF 和作用域清理。 |
| 原单路 Region+Depth | PASS，独立消费者收到 20 帧；`target/m3t-dual-camera-baseline/result.json`。 |
| 双路 Depth-only | PASS，240 帧；左右关节 RMSE 为 0.000339 / 0.006184 rad。主路无观测的 40 帧中仍跟踪运动，两个连续 20 帧区间均优于保持最后可见帧；`target/m3t-dual-camera-depth-v3/result.json`。 |
| 双路 Region+Depth | PASS，240 帧；左右关节 RMSE 为 0.009494 / 0.011381 rad；`target/m3t-dual-camera-region-depth/result.json`。 |
| 双路 extended | PASS，240 帧；覆盖 Region+Depth+Texture、每视角遮挡/轮廓渲染器、静态 seed 检测与初始化细化、soft constraints；左右关节 RMSE 为 0.006331 / 0.015235 rad；`target/m3t-dual-camera-extended/result.json`。 |
| 资源回收 | 上述端到端运行均 `children_reaped: true`，`forced_kill: []`。 |

双路 verifier 使用两个独立原生 provider 和非 identity 的合成外参，采集时间来自明确共享的合成 Device 时钟；默认按 capture 配对，所有输出均保留第二路身份、外参和双方完整时钟信息。两路交替将 RGB-D 图像填零，但流始终连续，验证的是观测缺失及恢复，不是物理出画、实物遮挡、热插拔或真实曝光同步。数值来自简化模型，不表示 SO-101 实机精度，也不是单路/双路精度对比 benchmark。

早期低采样配置（`model_divides: 0`、`model_points: 64`）未通过纯深度跟踪判据，保留在 `target/m3t-dual-camera-depth{,-v2}/result.json`。最终 verifier 使用正式模型采样参数 2 / 200 / 512，保留逐段运动与 hold-last 误差检查，未放宽通过条件。真实第二路相机的安装外参、时间关联、动态 pitch 双相机及 ARM64 本轮功能尚未实机验证。

复跑命令见 [README 双路说明](README.md#两路-rgb-d-相机可选)；运行 `scripts/verify_dual_camera.py` 时分别选择 `--profile depth-only`、`baseline`、`extended`，每次使用新的输出目录。

## 2026-10-26：Pitch 相机运动补偿接入

证据位于 `target/so101-base-reference-20261025/integration/`；板端构建记录位于 `~/rbt-work/manifests/pitch-integration-20261026/`，修正配置后的实机采集位于 `~/rbt-work/runs/pitch-integration-20261026/capture-06/`。capture-05 保留为错误配置的对照证据。

| 项目 | 状态 | 证据与范围 |
| --- | --- | --- |
| 定向软件测试与 ARM64 部署 | PASS | 本机定向测试通过；ARM64 最终 6/6（0.72 秒），覆盖最近样本、单样本、旋转组合、时钟/epoch、并发接收及原有帧/先验路径。当前二进制 SHA-256 为 `f4af6df362aa89f61e35cc4edb3d715e45634ae1445871d2ca7d588926d8bb67`。 |
| 真实 RGB-D / pitch TF / 双臂编码器 / M3T | PASS：链路 | capture-06 的 30 帧估计全部携带 nearest pitch 及 10 项实测 q1–q5 先验。反馈时间偏移范围 −41.615 至 +11.466 ms，绝对值中位数 13.097 ms；采用 HostMonotonic received best-effort，精度仍未知。 |
| 队列与资源退出 | PASS：本次窗口 | 初次联机暴露 GL 计算阻塞反馈接收、导致 pitch Control 队列满而退出的问题，现改为 activation-owned 独立接收任务及有界互斥缓存；最终窗口无 QueueFull，所有自有进程组已回收。 |
| 电机读写范围 | PASS：只读 | 两臂 `writes_enabled=false`；pitch 仅轮询寄存器，未下发目标、使能或力矩控制命令。用户认证后只启用了 can0 的 1 Mbps 通信。 |
| 轴向修正接入错误 | FIXED：配置一致性 | 最初预览采用绕 colour 光心的 180° roll，但 capture-05 错用了绕 depth 光心的 roll。设备 SDK 的 depth←colour 外参使 colour 光心偏移 28.238 mm。现按 `M_fixed=M_original E S E^-1` 修正 mount，保持 SDK 权威外参 E，恢复原预览的 colour 光心。只改 mount 配置，未拟合外参。推导与设备只读查询见 `integration/rotation-audit/`。 |
| 实拍模型对齐 / 空间精度 | NOT ACCEPTED | capture-05 同一冻结帧、同一编码器 FK 对照：修正前后深度残差中位数左 150.2→14.0 mm、右 162.2→85.0 mm；这些是包含遮挡和背景的图像诊断，不是真值位姿误差。右臂仍明显失配；固定 world 底座及强关节先验不能补偿错误安装/零位。capture-06 彩色图几乎全黑，仅接受链路结果；q6 未标定。 |
| 云台主动运动验证 | NOT RUN | 本轮保持实物姿态，只接收实时反馈。 |

同帧配置对照图及审计 JSON：`integration/rotation-audit/pitch-capture-05-same-frame-camera-audit.*`；修正后真实运行图：`integration/overlay/pitch-capture-06-three-panel.*`。原始失败窗口、错误配置、缓存/路径修复记录与光学朝向候选均保留。配置及最近样本策略见 [README](README.md)。

### 去掉新增外参的对照

按用户要求关闭 `camera_motion`，保留相机内参与 SDK D2C 对齐、相同关节先验和模型参数。`capture-07-no-extrinsics` 仅关闭运动/外参链，保持两底座固定；`capture-08-original-baseline` 使用原 `diagnostic-tracking.json`，同时恢复 `fixed_base:false`，完整恢复接入前的行为。两次真实运行各输出 30 帧、每帧 10 项先验，输出不含 camera_motion；未启动 pitch provider，电机只读，所有自有进程组已回收。

`integration/no-extrinsics/encoder-fk-comparison.*` 在 capture-05 同一明亮帧、同一编码器角度上对比当前外参链和旧 `base_in_camera` 初始化。投影深度残差中位数左 14.0→13.6 mm、右 85.0→44.3 mm；右臂投影改善，仍未完全重合。这是 FK 投影诊断；两次实时 M3T 运行使用不同图像，不能当成同帧追踪精度 benchmark。完整记录见 `integration/no-extrinsics/comparison-status.json`。新增对照配置为板端 `~/rbt-work/config/m3t/pitch-disabled-comparison.json`；原 pitch 配置保留。

## 2026-10-24：实机同窗口 RGB-D / 编码器采集

`target/m3t-so101-live-prior-20261024/STATUS.json` 记录本轮状态。Gemini 335（`CP15641000AT`）收到 30 对 RGB-D，同时从双臂原生 SO101 `StampedBytes` 接收 145 组状态和对应原始编码器读数。保存图像与关节状态共享 HostMonotonic source/generation；所选因果样本序号 123，早于图像接收时间 22,332,710 ns。采集期间所有关节读数范围均为零。该结果验证同一窗口中的接收时间关联，不代表曝光同步。

相机、驱动、Composer 与探针均由脚本退出并回收，进程组为空，未发送电机目标或配置写入。新增快照时间字段经 AidLux 原生 OpenCV 4.5.4 `FileStorage` 读取检查通过；未知不确定度用 `*_uncertainty_known: false` 和缺省数值字段表示。

**实机对齐仍未验收。** 两侧肘关节直接换算约 91.52° / 93.10°，超过模型 90° 上限，原始数据未夹紧。当前腕部几何候选无法同时解释腕部相机固定分支；近似 PCB 中心的独立检查也未通过，尚不能确定是对应点、姿态还是固定安装关系的问题。未用这些候选启动实机 M3T 追踪。第一组诊断图为 `target/m3t-so101-live-prior-20261024/pose-A-diagnostic.png`；第二姿态需待用户完成腕旋转调整后采集。软件先验功能的验收与该物理标定问题分别记录如下。

## 2026-10-24：SO-101 关节先验路径

关节先验的原生单元、原生软件集成和 ARM64 CPU-only 构建检查已通过。所有软件运行均使用合成 fixture；真实 SO-101 驱动映射、相机时间关联和机械臂精度仍未验证。

| 项目 | 状态 | 证据与范围 |
| --- | --- | --- |
| 原生模块单元测试 | PASS | `target/m3t-joint-prior-20261024/ctest-final.log`：5/5，0.71 秒；覆盖 frame source、model、app config、joint prior 和 joint-prior optimizer。 |
| 原生 SO-101 `StampedBytes` 先验集成 | PASS | `target/m3t-joint-prior-20261024/native-integration-final/result.json`：原生 Composer、RGB-D/关节状态流与 optimizer，55 帧，13 帧应用先验；前 10 个应用帧最大角度差为左 0.005649 rad、右 0.004851 rad。停止发布后最后 5 帧均为 stale 且无先验因子。配置显式采用 `received` best-effort 关联，`capture_time_associated` 为 false，不代表曝光时间同步。 |
| 集成进程清理 | PASS | 同一 `native-integration-final/result.json`：所有子进程已回收、进程组为空、`forced_kill` 为空。 |
| 先验默认关闭的基线 | PASS | `target/m3t-joint-prior-20261024/default-disabled/result.json`：独立原生消费者收到 20 帧双臂 baseline 输出，序列 0–19；子进程均回收且无强制终止。 |
| ARM64 CPU-only 构建及安装 | PASS | `target/m3t-joint-prior-20261024/arm64/REPORT.md` 与 `arm64/remote-logs/final/run.log`：GCC 11.4，5/5 聚焦 CTest（0.45 秒）；隔离 stage 安装成功，`rbt-m3t` 与 `rbt-m3t-initialize` 的 `--help` smoke checks 通过。未运行相机、驱动或机械臂。 |
| 实际驱动 service、相机时间关联和模型映射 | NOT_RUN | 合成软件 fixture 不验证真实驱动数据、相机曝光同步或实测模型映射。 |
| 实际相机与机械臂先验精度 | NOT_RUN | 需要使用真实硬件和经校准的显式关节映射进行验证。 |

需要重跑时，使用新建的输出目录：

```bash
ctest --test-dir build/m3t --output-on-failure -R 'frame_source|model|app_config|joint_prior'
python3 perception/m3t/scripts/verify_joint_prior.py \
  --build-dir build/m3t --composer "$RBT_PREFIX/bin/composer" \
  --output-dir target/m3t-joint-prior-verify-rerun-1
```

该 verifier 把 native Composer、RGB-D fixture、SO-101 状态发布器与 `rbt-m3t` 连接起来，检查先验路径和 stale 样本丢弃。输出目录必须尚不存在。上述软件验证只覆盖合成输入；此前下方的构建、DSP 与合成追踪记录保留原样，不代表真实机械臂精度。

2026-10-21：软件 pipeline 已通过；按用户要求，真实双臂的初始化和追踪效果留待后续实机验证。

| 项目 | 状态 | 证据与范围 |
| --- | --- | --- |
| 固定上游及集成构建 | PASS | M3T 固定提交的 30 个核心源文件、`rbt-m3t`、`rbt-m3t-initialize` 和测试程序均完成编译链接。 |
| C++ 模块检查 | PASS | `target/m3t-ctest.log`：3/3，覆盖像素所有权、配对、时钟/标定、FK、关节限位/跨 π 展开及配置/输出。 |
| URDF 转换器 | PASS | 6/6，包含实际外部 URDF 的只读转换；生成 10 links、9 joints、6 个活动关节、19 个视觉元素和 400,798 面。 |
| 原生软件 pipeline | PASS | `target/m3t-software-fixture-20261021-final/result.json`：渲染 RGB-D → 原生 RBT Color/Depth/CameraInfo → M3T Region+Depth → 独立原生 RBT 消费者，接收 20 批双臂结果。 |
| 软件轨迹及结果正确性 | PASS | fixture 每臂 2 links、1 个活动关节。检查完整 SE(3)、单位、限位、时间戳和未知不确定度。左右臂 RMSE 分别 0.01174 / 0.04999 rad，冻结零角度的基线均为 0.27051 rad；两臂都有可测运动。此数据不是实际 SO101 的精度。 |
| 进程清理 | PASS | 软件验证全部子进程已回收，无 SIGKILL；未操作电机。 |
| Gemini 335 RGB-D 输入 | PASS | `target/m3t-hardware-20261021-light/result.json`：精确序列号 CP15641000AT，60 对 640×480 图像、D2C/标定/去畸变、深度单位 0.001 m、最大采集时间差 121,000 ns。保留实际设备时钟。 |
| OpenGL/Qt 基础设施 | PASS | 上游 RendererGeometry::SetUp、Mesa OpenGL 4.6、OpenCV 4.10、PNG 写入及 Qt 窗口 smoke；不是初始化器点选/PnP 验收。 |
| 上游构建 overlay | PASS | 三处修正：未知 Body ID 的返回路径、Renderer 空 vector 引用、Linux GLFW/GLEW 的 X11 选择。checkout 保持干净，构建目录生成的副本使用精确文本匹配。 |
| 本地安装 | PASS | `target/m3t-install`；已安装程序可运行，`ldd -r` 无缺失库或未定义符号，不链接厂商相机 SDK 或 ROS。运行库来自当前本地依赖前缀，该安装不是可独立搬运的发行包。 |
| 初始化文件发布 | PASS | `target/m3t-initialization-publish-test.log`：新文件、已有文件、悬空符号链接、已有临时文件及 16 个并发写入者；不覆盖原文件。检查直接提取生产函数，未运行 GUI。 |
| 真实双臂初始化/PnP | NOT_RUN | 用户决定先跑软件 pipeline，后续提供实际关节角和场景对应点。未生成虚构的实测初始位姿。 |
| 真实 SO101 追踪精度/遮挡恢复 | NOT_RUN | 软件 fixture 不能代替真实双臂；没有自动检测或丢失重定位的实现/精度声明。 |

运行方式见 [README](README.md)。本地集成构建使用 phase4 GCC/Ninja 和现有 RBT 0.8 SDK；此模块未修改 RBT 核心、协议、IDL 生成器或手写 unsafe。第三方完整许可和 overlay provenance 见 [NOTICE](NOTICE.md)、[UPSTREAM.toml](UPSTREAM.toml) 与 `licenses/`。


## 2026-10-23：ARM64 M3T 0.2.0 GPU-renderer 验证

此轮在 AidLux ARM64 设备上针对合成双臂 fixture 完成；不覆盖真实相机输入或实物机械臂精度。tracker 和 fixture 都实际使用 Mesa Zink / Turnip Adreno 740，OpenGL 4.6。编译、安装完成，`build-test-install.log` 中 3/3 CTest 通过（0.49 秒）。

| Profile | 状态与检查 | 关节结果 | 整次运行时间 |
| --- | --- | --- | --- |
| baseline | PASS；Region+Depth，20 批，序列 0–19 | RMSE 左/右 0.00926 / 0.05088 rad；两臂都有可测运动 | 7.153 秒 |
| extended + reinitialize | PASS；Region+Depth+Texture、modeled occlusion、region checking、YAML metadata、soft constraint；显式 SIGHUP 重初始化事件通过；20 批 | RMSE 左/右 0.01088 / 0.01428 rad；两臂都有可测运动 | 6.833 秒 |
| hard constraint | PASS；20 批 | 输入各臂移动 0.2112 rad，输出运动约 2e-8 rad，RMSE 与冻结零角基线相同；这是约束生效的专用检查，不表示精度提升 | 6.765 秒 |

以上时间是端到端运行时间，不据此推导 FPS。三个结果 JSON 均记录实际 renderer、`children_reaped: true` 和空 `forced_kill`。证据位于 `target/aidlux-m3t-0.2.0-validation/{build-test-install.log,baseline-gpu/result.json,extended-gpu-reinitialize/result.json,hard-constraint-gpu/result.json}`。

本环境验证了 CPU ORB；BRISK/SIFT 编译通过，但未分别进行运行时验证。CUDA 与 OpenCV contrib 不可用，CUDA ORB、DAISY/FREAK 未验证。Gemini 335 实机采集、真实 SO101 初始化/精度/性能均为 NOT_RUN。上游数据集 benchmark 示例未移植；每个约束只作用于一个 arm 内的 link pair，不支持跨 arm 或跨独立 optimizer coupling。此验证不表示完整自动检测器或全 GPU optimizer 已接入。

## 2026-10-23：ARM64 M3T 0.3.0 Hexagon DSP ORB

此轮验证使用 AidLux ARM64 / Adreno 740 设备。Renderer 与 GPU fixture 均报告 Zink / Turnip Adreno 740；`hardware: NOT RUN` 表示 RGB-D 和双臂跟踪数据来自合成 fixture。DSP skeleton 在 CDSP 实际加载并执行，因此以下结果确认设备 DSP 后端功能，但不代表真实相机或 SO-101 验收。

| 项目 | 结果 | 范围 |
| --- | --- | --- |
| DSP-off build / software | PASS | ARM64 6/6 CTest；baseline 与 extended 合成 CPU texture profile 各 20 批。 |
| DSP-on build / install | PASS | ARM64 7/7 CTest；安装版本、runtime hash 与五个 DSP license 文件检查通过。 |
| V73 CDSP 独立 ORB 测试 | PASS | 七个提取案例、四个批量匹配案例；300×300 descriptor pair 返回 600 个邻居，每个非空矩阵操作一次 RPC，无失败。七个提取案例与 portable CPU core 的 keypoint 全字段及 descriptor 字节一致。 |
| CPU extended fixture | PASS | 20 批；Zink/Turnip Adreno 740；整次运行 6.716 秒；子进程均回收且无强制终止。该 profile 不执行 DSP。 |
| DSP extended + SIGHUP reinit fixture | PASS | 20 批；DSP audit 记录 112 次提取事件/RPC / 7,513 个特征、88 次匹配事件/RPC / 977 queries / 6,292 train / 1,954 neighbors，即每个非空 operation 一次 RPC；重初始化请求与事件通过；整次运行 13.855 秒，子进程均回收且无强制终止。 |

fixture 的两组整次耗时包含不同工作，不是 FPS，也不是性能对照。真实 Gemini 335 RGB-D 采集、SO-101 初始化和物理追踪精度仍为 **NOT_RUN**。

独立设备 microbenchmark 在 AidLux V73 上以 200×200 图像、300 个特征、3 层、5 次预热和 20 次采样执行，数值为毫秒 p50 / p95：

| 操作 | DSP facade | portable CPU core | OpenCV CPU |
| --- | ---: | ---: | ---: |
| ORB 提取 | 45.3244 / 45.4193 | 10.6742 / 14.1625 | 1.93681 / 5.7585 |
| 300×300 匹配 | 2.32049 / 2.49559 | 0.174942 / 0.185931 | 0.291188 / 0.489879 |

这组 microbenchmark 中 DSP facade 的提取和匹配均慢于所测 CPU 路径，不支持加速结论。DSP 数值实现使用固定点重采样/配额、Gaussian rounding 和 CORDIC，不能声称逐位兼容 OpenCV ORB；设备案例中的精确相等对象是 portable CPU core。主机仍负责 ROI、Lowe ratio、几何检查与 M3T optimizer，optimizer 在 CPU 上；本轮未实现 HVX 向量路径或 NPU。

M3T skeleton 为 Hexagon V68 交叉编译，在 V73 CDSP 上验证；这不证明它兼容所有 Hexagon 固件。最终 skeleton 和 ARM64 stub 的 SHA-256 如下：

| Artifact | SHA-256 |
| --- | --- |
| `librbt_m3t_orb_skel.so` | `10d2e3ecfebacb6b578d665b4e7d1118094fe2fcdd105d2e532019b2c82349d5` |
| `librbt_m3t_orb_stub.so` | `d229c3f5e0e63d6d3e3bcb619c675d48c527b01a4806a6b16d580d3a4ab65b4e` |

最终设备结果、CPU/DSP profile JSON 和验证证明随源码包放在 `verification/`。完整构建、安装和运行原始日志仍保存在源码仓库的 `target/aidlux-m3t-0.3.0-dsp-validation/final/`，不随包分发。最终独立设备 oracle 与 microbenchmark 输出为 `target/aidlux-m3t-0.3.0-dsp-validation/phase3/evidence/final-device-benchmark.stdout`，包含全部设备 oracle 检查；较早的 `target/m3t-hexagon-bootstrap-20261023/evidence/production-device-test.stdout` 是基线历史输出，不是最终完整结果。安装包已包含 `compiler-rt-LICENSE.TXT`、`fastrpc-LICENSE.txt`、`opencv-orb.txt`、`qaic-LICENSE.txt` 与 `qaic-NOTICE`。

### 2026-11-13：左臂刚性 ArUco 标定候选（未通过）

- aidlux 本机 8098 / SSH 转发 18898 已切换至 `calibration/aruco_web.py`；使用当前官方软鳍模型、左臂 v2、`DICT_4X4_50` ID 1，用户确认黑色外框 29 mm。原双相机数据保留，新数据独立保存。
- 软件验证：本机 calibration 65 项测试通过；aidlux `test_aruco_web` + `test_handeye` 共 31 项通过。主相机识别、同号拒绝及左臂 v2 寄存器/反馈均已实时检查；无电机运动写入。
- 实机数据目录：`/home/aidlux/rbt-work/calibration/aruco-left-v2-20261113-110031`。16 姿态结果 `result-1794539567474424926.json` 为 `REJECTED_FIT`：训练 RMS 5.3101 px，最大帧 11.5671 px，留一姿态最大 16.0405 px；运动多样性、满秩和正深度通过。**没有采用为 M3T 初始化。**
- 所有样本的独立单帧 PnP 误差 0.0296–0.6631 px；保存区间五关节反馈最大变化 0.26374°。用户确认贴码始终固定硬壳、未碰动。该信息不能排除反馈之外的机械变形或相机误差。
- 离线五初值拟合收敛到同一 5.3101 px 结果。仅诊断的内部三关节零偏扩展拟合将训练 RMS 降至 3.3822 px，但留一姿态仍最大 11.9612 px，不能据此接受或修改运行配置。原始数据、URDF、v2 和运行外参未被该诊断修改。

### 2026-11-13：按用户要求替换 aidlux 左臂静态安装边

- 已将当前 12 姿态候选（排除 10/11/15/16）转换到现有父帧并替换 `chest_camera_center -> left_base`；此 `left_base` 使用当前官方软鳍 URDF `base_link` 定义。右臂和其他九条边均语义不变。
- 已备份旧 YAML，来源与变换审计保存在 aidlux `/home/aidlux/rbt-work/runs/static-tf-left-new-20261113/` 和静态 TF 包的 `calibration/left-official-soft-fin-20261113/`。主配置路径 `/home/aidlux/rbt-work/src/static_tf_rbt/config/robot.yaml`，配置链接指向同一文件。
- 转换使用现有静态安装链、现场验证一致的 Orbbec 出厂 depth<-color 变换及 `Ry(-pitch)`。**采集未逐帧记录云台角度，按固定 −61° 假设换算**；当前控制目标及现场读数接近 −61°，不构成采集时角度的独立证明。候选严格拟合门槛仍未通过，部署身份为用户授权的实验初值。
- 配置检查 10 条边通过；使用已有 Composer 启动静态发布器（未启动云台电机节点、未发送电机目标）。真实原生 RBT 查询已读回新 `chest_camera_center <- left_base`、完整静态链 `camera_pitch_base <- left_base` 及保持不变的右臂安装边。矩阵回代误差 <1e-12。
- 当前仅验证静态发布及数学组合，尚未做新外参的 M3T 实机跟踪验收；云台动态节点仍由既有运行方案另行接入。

### 2026-11-13：第一步只读模型叠加验证已上线

- `aruco_web.py /verify` 接入 `overlay_preview.py` / `model_overlay.py`。固定使用已部署 12 姿态候选及对应审计，核对静态配置、报告、URDF 和 v2 来源哈希；不使用 M3T 或标记观测修正模型姿态。
- 每轮只读查询云台状态、读取前后两次编码器反馈并包围新主相机帧，移动超过 0.3°、云台偏离 −61° 超过 0.5°、来源变化、反馈/图像错误或缓存过期时不显示叠加。软件时间不代表曝光同步。
- 显示八个刚性 link 的网格外轮廓；省略尚未标定开合映射的活动夹指及活动软鳍。每个 link 独立投影，不做跨 link 隐面消除，页面明确提示遮挡轮廓可透视显示。
- 本机和 aidlux 定向检查：模型 FK/固定子 link/相机投影/非法输入 3 项、预览来源/云台角度/过期保护 4 项、原 ArUco 采集 6 项，共 13 项通过。浏览器实际确认实时图像、关节反馈、原图切换及保存验证画面。
- 首帧记录 `/home/aidlux/rbt-work/calibration/aruco-left-v2-20261113-110031/verification/view-1794542041839876041/` 包含原图、叠加图和完整反馈来源；云台约 −61°，实时预测码框误差约 15 px。该姿态接近旧第 16 组，不能作为独立新姿态验收。网页已保持开启供用户手动摆新姿态验证，未发送电机目标。

### 2026-11-13：真实 M3T 叠加与实时五关节软先验

- 实验原生包支持单臂或双臂，输出原子预览包：实际 estimate JSON 与该次跟踪输入的 BGR8 图像一同发布。网页用每个 link 的真实输出矩阵画橙色轮廓，青色保留原始 v2 编码器 FK；没有从跟踪角度伪造输出位姿。隔离 stage 安装不替换原全局 M3T，ARM64 原生六项 CTest 已通过。
- 原生实验入口为 aidlux `runs/m3t-left-overlay-20261113/run-m3t-experiment`；本轮会话 `session-rbv12wip` 使用 Region + Depth、固定 base、当前官方软鳍模型和用户接受的 12 姿态实验初值。
- `joint_feedback_feed.py` 独占两个真实串口，检查两臂 v2 六电机的 Homing_Offset / Min_Position_Limit / Max_Position_Limit，只读采样并发布 `rbt/so101/v2-m3t-readonly/state/v1`。不创建命令订阅器、不调用 DriverService.run、不写目标、力矩或电机配置。任一读取失败即退出，缓存自然过期。网页从同一份原子反馈缓存读取，不再重复打开串口。
- 左臂五个角度按 π/180 转为 rad、零 offset，作为每次优化的软先验，标准差 0.1 rad（约 5.7°）；夹爪百分比不作为弧度先验。先验 required，使用 received 软件时间、最大年龄 100 ms；没有宣称曝光同步。网页同时核查实际输出 applied、5/5、关节名称、controller 与年龄。
- 真实诊断读回 `status=applied`、`prior_count=5`、`rejected_mappings=[]`；记录帧 2640、反馈序号 8251、年龄 8.216 ms。暂停只读发布器后旧反馈超过门槛，原生输出帧号在 908 停止更新；恢复发布后推进至 945 并再次应用全部先验（2.800 ms）。全程没有机械臂运动命令。
- 本机 calibration 原有加新增测试共 84 项通过，随后新增一项实际先验诊断门槛测试通过；aidlux 定向反馈/发布器/预览 11 项通过。故障测试覆盖读失败无新发布、资源关闭、错误寄存器、未知缓存文件及先验身份/年龄/数量异常。
- 实机审计、启动脚本、源码哈希和中断恢复证据在 aidlux `/home/aidlux/rbt-work/runs/m3t-joint-priors-20261113/`。已保存 `verification/view-1794558797219515391/` 的原图、叠加图和实际先验诊断。预览原采集程序达到 86400 秒上限后，启动独立新预览采集；未覆盖原始标定数据。
- 当前姿态 ID 1 不可见，无法提供独立码框误差或视觉精度改善结论。网页能显示两种轮廓，仍需用新姿态验证贴合；实时先验生效不等于外参标定或跟踪精度验收通过。

### 2026-11-13：恢复双侧夹爪显示

- `model_overlay.py` 不再省略活动夹指和活动软鳍。青色显示将 v2 的夹爪开度 0–100% 线性映射到名义 URDF 闭合 0 rad 和张开上限 1.74533 rad；开度及名义端点写入 receipt，缺省输入的闭合假设单独标记。
- 该映射只用于显示，没有测得百分比到 URDF 角度的对应关系：v2 raw 范围 2025–3460 约占电机 126.15°，不等于 URDF 的张开范围；没有把 URDF −10° 下限当成已验证的闭合端点。没有改 URDF、v2、电机限位或五关节先验。
- 橙色恢复活动夹指/软鳍的实际 M3T per-link 输出，保持原始矩阵，不套用青色显示映射。网页显示夹爪百分比；前后反馈开度变化超过 0.5% 时不显示该轮叠加。
- 本机模型投影 8 项、预览 4 项通过；aidlux 模型投影/预览/M3T 包验证共 17 项通过。已重新启动网页及其拥有的 M3T 实验，真实 receipt 确认两路轮廓均渲染 10 个 link、无省略项，五关节先验仍全部 applied。审计与备份在 aidlux `runs/m3t-gripper-display-20261113/`，保存画面 `verification/view-1794559356175645836/`；当前实机开度 0%，中间/张开端点仅做软件测试，未控制机械臂开合。

### 2026-11-13：独立显示图层开关

- 网页新增青色编码器模型、橙色 M3T（含其预测码框）、粉色预测码框、绿色实测码框四个独立开关，并保留原图总切换。开关只影响绘制，跟踪、反馈读取、过期保护和误差计算继续运行；保存 receipt 记录显示选择，切换时清除旧图，避免沿用旧图层。
- 本机预览 6 项、ArUco 网页 6 项通过；aidlux 预览/投影/M3T 包/ArUco 网页共 25 项通过。真实浏览器逐一关闭并恢复四个开关；全部关闭的 `view-1794559734035453317` 中 overlay.png 与 main.png 逐像素相同。保存帧 487 后实际 M3T 推进至 800，五关节先验仍 applied 5/5。当前已恢复四层均显示供用户选择。
- 部署备份、源码哈希、实时状态及全部关闭证据在 aidlux `runs/m3t-layer-toggles-20261113/`。未发送机械臂或云台运动命令。

### 2026-11-13：撤销 URDF 上限缩放，使用 v2 编码器行程并补齐夹爪先验

- 之前把 100% 开度对应 URDF 上限的显示映射不代表实物标定，现已撤销。`gripper_calibration.py` 从当前 v2 读取 raw 2025–3460 和方向，按已安装驱动的 4095 分辨率计算 1435 ticks = 126.153846°；显示与第六项先验统一用该比例 0.02201799979439 rad/%，不使用 URDF 上限作为测量比例。
- 闭合实物端点暂赋模型零角；该参考和 URDF 零位的安装对齐仍未独立量测，不能声称 v2 已确定 URDF 零位。初始化也使用实际反馈开度，不再要求近闭合。前五项 σ=0.1 rad 不变，夹爪 σ=0.01 rad；原生预览必须实际 applied 6/6、反馈不超过 100 ms。
- 每次实验创建独立 robot.json，单独把夹爪跟踪上限容纳到真实行程 126.153846°，其他关节数据不变；原模型、URDF、v2 和电机限位未修改。原生加载器要求模型目录内相对 OBJ 路径，运行副本携带源网格的逐字节快照，哈希及限位差异写入 runtime-model-audit.json。首次绝对网格路径尝试被原生加载器拒绝，已修复并重新通过实机启动；未放宽加载器路径检查。
- 本机物理行程换算 3 项、M3T 预览 8 项、模型显示 9 项、预览 6 项通过；aidlux 相同 26 项通过。实机会话 `session-i4xspmo_` 确认 applied 6/6，夹爪先验 0 rad，实际估计约 0.03°；此前同一闭合反馈下未加夹爪先验时约 57°。这只验证闭合约束恢复，不代表整臂贴合精度通过。
- 审计、备份和实际诊断在 aidlux `runs/m3t-gripper-prior-20261113/`；保存原图/叠加/receipt 为 `verification/view-1794561486377497521/`。保留用户仅开启青色的图层选择。没有发送机械臂/云台运动、电机目标或配置写入；完全张开与中间开度仅做软件换算测试，未做物理开合验收。


### 2026-11-13：继续新双目标定 TF 激活及真实双相机连接烟测

- 原 chat 已有用户确认新标定采集俯仰为 −61°。本次审计报告 SHA-256 为 `8ed5abee6f7f6dace646a4afe7a6a76ecbf8c419e45c7e5b0721c22cf2225331`，25 对、stereo RMS 0.252679 px，训练最大交叉重投影 1.525906 px，留出最大 2.244593 px；状态保留 `CANDIDATE_REQUIRES_PHYSICAL_VERIFICATION`。
- 新静态配置 SHA-256 为 `2bc8ab764e6d407c66a1f5280f1a13c0927642eb00f812c6656636ccca99af31`，只改云台底座与左臂安装两条边。−61° 组合复现新矩阵，同时保留原主相机下的左臂手眼初值；数值差均低于 1e−12。原生快照和恢复后的动态 pitch 链查询均通过。这是换算与消息链验证，不是空间精度测量。
- 本地 Orbbec 7/7、M3T 9/9，通过原生双相机软件 fixture（240 estimates）；本地 M3T 首次因 libopenblas 运行路径缺失失败，设置现有依赖路径后通过。AidLux 原 M3T 9/9。板端源码包含 namespace、Gemini2 独立深度 profile、硬件 D2C 等后续功能，不能用本地旧分支整体覆盖；现以独立兼容补丁增加 Secondary，Release CTest 16/16。
- 实际 Gemini335 + Gemini2 双路 M3T 完成 30/30 estimates：每帧保留第二路 serial/frame、新外参和六项先验，跨视角 capture 时间差 0.663–17.146 ms，最大先验年龄 32.879 ms。测试时俯仰约 −62.2°，只验连接与 provenance，不作为 −61° 标定精度验收。测试结束原 Gemini2 runtime/launcher 已恢复，并实际读到 3 帧新深度图。
- `tools/pitch_tf_bridge` 只 GET 现有角度网页，发布原生 `camera/PitchTf`，不接触 CAN、不注册控制 RPC。解析/几何/原生 wire 与 lookup、HTTP 故障不复用旧值、CLI 时长与信号三项 CTest 通过，主代理重跑 3/3。动态时间戳为 HTTP 接收近似值、uncertainty 未知，源缓存最多 500 ms，不代表曝光同步或已通过运动补偿验收。
- 网页 TF 场景已使用 2026-11-12 新版，6 项网页测试通过。叠加页恢复新鲜 RGB 订阅并保留原采集 session 的内参绑定；恢复流的实际 CameraInfo 与原 session 逐项比较一致，新 provider activation/session 单独记录。用户原来的青色单层选择保留。当前俯仰约 −62.14°，原 −61° ±0.5° 门槛继续阻止绘制，网页 M3T 保持停止。
- 完整记录、备份与证据索引见 [`deploy/20261113-stereo-tf-and-dual-camera.json`](../../deploy/20261113-stereo-tf-and-dual-camera.json)。没有发送机械臂或云台运动指令；原始标定数据未改，左臂 solver 拒绝状态和其他角度精度待验证的边界均保留。


### 2026-11-13：用户回到 −61° 后的静止画面对照

- 读取实际云台反馈约 −60.98°，原 ±0.5° 门槛通过，新鲜编码器叠加恢复。原图和反馈保存为 `verification/view-1794571375524364236`。
- 按当前六项反馈启动单主相机 M3T，对照帧保存为 `verification/view-1794571494152538597`。六项先验实际 applied，但肘部相对编码器仍偏约 −25°，腕弯约 +35°；肩抬估计触及 110° 模型上限。此差异不能作为已修正成功的证据。
- 由同一保存帧的实际 M3T per-link 矩阵与原始反馈 FK 生成对照图，未改变用户网页的青色单层选择。保存后停止实验 M3T，保留新鲜青色叠加；没有发送电机命令。
- 夹爪 ID 1 不可见，未取得独立码框误差。本次没有修改标定、模型或先验参数；需在码可见的静止新姿态继续排查左臂安装／零位／模型对应关系。


### 2026-11-13：将云台运行原点落实为 Fusion 截图的电机中央顶面圆心

- 用户截图点 1 为 DM3510 中央凸起顶面圆心（face #81298），点 2 为 Type-C 侧相机背部 M3 孔口；CAD 复现点间距离 60.394747 mm，分量绝对值 56.699960 / 20.800000 mm。组合官方 RGB 名义原点后，用户 Fusion 轴下为 `[23.20, 0, -48.59]` mm，换入既有 runtime 零位轴后为 `[48.59, -23.199960344, 0.000008868]` mm。
- 原机最近轴点原点替换为用户所选实体原点，同时把名义 FOV 杆臂 47.52 mm 改为官方孔位尺寸推算的 48.59 mm。两条固定边配套换算，保留 −61° 实测双目姿态。全部静态旋转、设备出厂深度/彩色外参、左右臂安装边不变。
- 独立 48 项离线检查通过，轴向原点换算在全角度可相消。板端配置检查为 10 条边；相机固定边、机身到云台边、带实际 pitch 样本的完整原生 TF 查询均与候选矩阵一致。当前约 −61.024° 时，新旧光心位置仅差 0.000446 mm；这是模型差异计算，不是实物精度结果。网页 6 项检查通过，实际叠加页恢复 ready，保留用户青色单层；保存帧 `view-1794579392957804623`。
- 已激活配置 SHA-256 为 `a40eda84cf77c53a394f3b98ea87a01ca0ab7e036edb2d55d4ab54e62faec8b3`。完整来源和回滚见 [`deploy/20261113-gemini335-cap-origin.json`](../../deploy/20261113-gemini335-cap-origin.json)。没有发送运动指令或替换相机驱动；该更新不宣称解决原 M3T 关节偏移，零位、回差和独立多角度精度仍待验证。

## 2026-11-15 CAD mount refinement experiment

Opt-in fixed-camera CAD base soft constraints and exact encoder joint transforms were
built locally and on AidLux. Targeted native CTest programs passed 3/3 on both hosts;
Python checks passed 65 locally and 47 in the board app environment. Eight selected
live samples retained 12 applied measurements, zero reported joint deviation, and
rotation orthogonality/determinant errors below 1e-6 after quaternion projection.

Per-link visual weighting is implemented and a 5x rigid-gripper trial was exercised.
That trial remained visibly mismatched and produced a large right mount rotation;
it is retained as an experiment, not the selected default. The active page uses CAD
base refinement with default visual weights. Neither end-effector fit nor physical
mount correction is accepted. Evidence and rollback are recorded in
`deploy/20261115-m3t-cad-base-refinement.json`; recordings remain under ignored `target/`.
