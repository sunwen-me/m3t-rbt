# RBT M3T 双臂跟踪

用户于 2026-11-12 确认：后续 SO-101 M3T 统一使用 [`so101_lerobot_soft_fin.urdf`](models/so101_lerobot_soft_fin/so101_lerobot_soft_fin.urdf)，即 LeRobot 官方 SO101 加此前软鳍夹爪的版本。模型生成、初始化、配置及跟踪均以此为准；未经用户明确要求，不得切回旧数字关节模型。此确认针对模型选型和显示几何，不代表实机标定或部署验收。

双从臂的舵机校准统一使用用户确认的 `my_bimanual_follower_v2_left.json` / `my_bimanual_follower_v2_right.json`。AidLux 的 `config/so101/dual_follower_{left,right}.json` 和 `config/so101-control/calibration/dual_follower_{left,right}.json` 均指向这版 v2；旧版校准已移除。私有校准 JSON 不进入 Git。舵机校准与 M3T 相机初始位姿是两类数据，前者不能替代后者；夹爪状态的百分比也不能直接当作模型弧度。

RBT M3T 0.3.0 是可选原生模块，集成固定提交的 DLR-RM M3T，并继续使用 RBT C++ SDK 0.8.0。默认用 Region 与 Depth 模态，根据实测初始化文件跟踪两臂；Texture 可选。输出是 link 位姿和关节角估计，不是识别置信度。模块不会自动识别机械臂或在目标丢失后搜索重定位。

SO-101 M3T 只使用 [`models/so101_lerobot_soft_fin/README.md`](models/so101_lerobot_soft_fin/README.md) 所述的官方来源 soft-fin 模型；它保留 LeRobot SO-101 follower 的关节名、轴和原点；按当前 v2 行程，将 `shoulder_lift` 与 `wrist_flex` 的模型限位扩为 ±110°，`elbow_flex` 上限扩为 100°、下限保留 −1.69 rad；按用户要求兼容 v2 全量程，将 `wrist_roll` 扩为 ±180°，其余沿用来源模型。旧数字关节模型目录、压缩包及 AidLux 数字关节 wrist mapping 配置已删除，仓库只保留这一套模型。初始化必须来自实际 RGB-D 快照和当时测得的关节角，不能用猜测值或示例姿态代替。

## 构建

需要 RBT C++ SDK、C++20、Eigen、OpenCV 4（core、imgproc、imgcodecs、calib3d、highgui、features2d）、GLFW、GLEW、OpenGL 和 OpenMP。上游固定版本和 overlay 见 [`UPSTREAM.toml`](UPSTREAM.toml)。引导脚本会准备固定提交，不安装系统包：

```bash
./perception/m3t/scripts/bootstrap_native.sh
export RBT_PREFIX=/path/to/rbt-sdk
cmake -S perception/m3t -B build/m3t -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON -DCMAKE_PREFIX_PATH="$RBT_PREFIX"
cmake --build build/m3t --parallel 4
ctest --test-dir build/m3t --output-on-failure
```

M3T 跟踪器需要 OpenGL context，即使不开 `--preview` 也需要可用的显示环境。纹理描述器、CUDA 和 OpenCV 可选组件的限制见下文。

可选 Hexagon DSP ORB 的构建、运行时库路径和验证方式见 [`dsp/README.md`](dsp/README.md)。它不改变默认构建；固定 QAIC 生成文件已随源码提供，常规 DSP 构建无需安装 QAIC 或 Haskell。跨编译工具链仍须由构建环境提供。

## 生成模型与实测初始化

把生成的模型、快照和实测初始化放在 `target/`，避免将私有校准 JSON 带入源码包。下面使用官方来源 soft-fin 模型；它只生成 URDF 与 provenance，不会伪造 M3T 初始化：

```bash
export M3T_DATA="$PWD/target/so101-lerobot-soft-fin"
mkdir -p "$M3T_DATA"
python3 perception/m3t/models/so101_lerobot_soft_fin/build_model.py \
  --output "$M3T_DATA/model"
python3 perception/m3t/scripts/prepare_robot.py \
  --urdf "$M3T_DATA/model/so101_lerobot_soft_fin.urdf" \
  --output-dir "$M3T_DATA/robot"
```

`prepare_robot.py` 的输出目录必须尚不存在。旧模型生成的 initialization、base pose 和 encoder offset 不能只改文件名后复用；需要按当前模型、实际 RGB-D 快照和实测关节角重新生成并复核。

启动 Composer 和已对齐的相机，在新目录中采集 RGB-D 快照：

```bash
# 终端 1
composer

# 终端 2
rbt-orbbec-camera run --serial "$CAMERA_SERIAL" --align-depth-to-color

# 终端 3
rbt-m3t probe --provider orbbec335 --count 30 --duration-sec 10 \
  --snapshot "$M3T_DATA/snapshot" --preview
```

快照的 `calibration.json` 同时保留深度/彩色采集时间与 `received_ns`、`received_clock_kind`、`received_clock_source`、`received_clock_generation`。只有接收时钟身份和 generation 相同，才可按接收时间关联关节状态；这不代表相机曝光与编码器读取同步。`capture_uncertainty_known` / `received_uncertainty_known` 为 false 时，对应 `*_uncertainty_ns` 字段缺省，精度仍为未知。

按快照时机械臂的实际状态填写每个 arm 的当前关节角（弧度），再运行交互初始化器。输出文件不会被覆盖：

```bash
rbt-m3t-initialize \
  --model "$M3T_DATA/robot/robot.json" \
  --snapshot "$M3T_DATA/snapshot" \
  --output "$M3T_DATA/initialization.json" \
  --joint-angles "$M3T_DATA/joint_angles.json"
```

仅当两臂在快照时所有关节确实为 0 rad，才可使用 `--neutral`。初始化 UI 中，每臂选至少 6 组分布良好的模型 3D 点与快照 2D 点后按 Enter 求解。初始化记录了实测底座位姿、关节角、相机序列号和坐标帧；这些相机信息必须与运行时一致。

## 配置与运行

新官方来源模型的基础示例是 [`configs/so101_lerobot_soft_fin.example.json`](configs/so101_lerobot_soft_fin.example.json)。从仓库根目录直接使用时，其中 `../../../target/so101-lerobot-soft-fin/...` 路径指向前面生成的模型、实测 initialization 和缓存目录。若将 config 复制到 `target/`，把这三条路径分别改为 `robot/robot.json`、`initialization.json` 和 `cache`。该初始化文件只有在按快照时的实测关节角运行 initializer 后才存在。

新模型保留官方 fixed `gripper_frame_link`，它表示官方名义 TCP，不代表软鳍接触点标定。新增几何的局部坐标根据共同 CAD 锚点映射；该映射不携带旧模型的 arm zero 或限位。相机模块与 hand-eye 分支不在此模型中。

原兼容模板 [`configs/so101_dual.example.json`](configs/so101_dual.example.json) 与扩展示例 [`configs/so101_dual.extended.example.json`](configs/so101_dual.extended.example.json) 仍可用于其他已准备的模型。扩展示例中的模型、初始化路径是占位值，不附带物理初始化文件。复制示例与 YAML 到 `target/` 后，改写相对于新配置文件的模型、初始化和缓存路径：


```bash
cp perception/m3t/configs/so101_dual.extended.example.json "$M3T_DATA/config.json"
cp perception/m3t/configs/*.example.yaml "$M3T_DATA/"
python3 - "$M3T_DATA/config.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as source:
    config = json.load(source)
config["robot_model"] = "robot/robot.json"
config["initialization"] = "initialization.json"
config["cache_directory"] = "cache-extended"
with open(path, "w", encoding="utf-8") as output:
    json.dump(config, output, indent=2)
    output.write("\n")
PY

rbt-m3t run --config "$M3T_DATA/config.json" --count 100
```

这份 config 的 metadata 路径仍相对于 `config.json`，因此复制的 YAML 与它放在同一目录。初始化路径必须由上面的实测初始化器生成，示例文件不会伪造它。

配置继续使用 `schema_version: 1`。模型、初始化、缓存和 metadata 路径都相对于配置文件；metadata 必须是可读普通文件。嵌套对象和配置字段采用严格校验，未知字段会报错，未写字段保留默认值。

| 配置对象 | 字段与默认值 |
| --- | --- |
| `modalities` | `region: true`、`depth: true`、`texture: false`；至少启用一种。 |
| `occlusion` | `measured: true`、`modeled: false`、`region_checking: false`、`image_size: 512`（范围 64–2048）。 |
| `metadata` | 可选 `region`、`depth`、`texture`、`optimizer`、`refiner` upstream YAML 路径。 |
| `initialization_flow` | `static_detector: false`、`refine: false`。静态检测器以实测 seed 为起点；Refiner 可选用于启动时细化。它们不提供自动识别或全场景初始位姿搜索。 |
| `visualization` | `correspondences: false`、`optimization: false`、`results: false`；控制跟踪循环的上游可视化调用。 |
| `constraints` | 最多 64 项，每项含 `name`、`arm`、`link1`、`link2`、`type`（`hard`/`soft`）和 `metadata`。arm 必须是一个已配置 arm，两个 link 必须是模型中的不同 link。 |
| `joint_priors` | 可选的 SO-101 `StampedBytes` 关节状态先验；未配置时关闭。对象存在时默认启用，`required: false`、`initialize_from_measurement: true`、`time_basis: "capture"`、最大年龄 50 ms、缓冲容量 128。 |
| `camera_motion` | 可选的 pitch 相机运动补偿；未配置或 `enabled: false` 时关闭。启用时要求所有 arms 为 `fixed_base: true`，并为每个 arm 提供固定的 world 底座。 |

### 两路 RGB-D 相机（可选）

在现有配置中加入 `secondary_camera`，可让两台相机的观测共同参与同一个物体的优化。每个视角使用独立的相机、模态和渲染器，共享物体几何模型及原有的每臂 optimizer；不是先分别跟踪再平均位姿。完整配置形状见 [`configs/so101_dual_camera.example.json`](configs/so101_dual_camera.example.json)。原来的 `so101_dual` 指双臂，不代表双相机。

```json
{
  "secondary_camera": {
    "provider": "orbbec-secondary",
    "serial": "REPLACE_WITH_SECONDARY_SERIAL",
    "coordinate_frame": "secondary/color_optical",
    "camera_in_primary": [1, 0, 0, 0,
                          0, 1, 0, 0,
                          0, 0, 1, 0,
                          0, 0, 0, 1],
    "maximum_skew_ms": 20,
    "time_basis": "capture"
  }
}
```

上面的 identity 外参只是字段形状示例，必须换成实测的 `T_primary_color_optical<-secondary_color_optical`，平移单位为米，矩阵按行排列。两台相机须分别启用 D2C，提供对齐的 RGB8 + U16 深度及各自标定；内参、分辨率与深度单位可以不同。第二台的 provider、序列号和光学帧须与第一台不同。启动同一个 Composer 后，在不同终端启动两路驱动和跟踪器：

```bash
rbt-orbbec-camera run --serial "$PRIMARY_SERIAL" \
  --component-id orbbec-primary --align-depth-to-color

rbt-orbbec-camera run --serial "$SECONDARY_SERIAL" \
  --component-id orbbec-secondary --frame-prefix secondary --align-depth-to-color \
  --secondary-camera

rbt-m3t run --provider orbbec-primary --config "$M3T_DATA/config.json"
```

主相机的初始化文件仍须匹配主相机序列号和光学帧。第二台驱动的 `--secondary-camera` 发布独立的 Secondary 能力类型；当前 SDK 按能力类型绑定，单靠不同组件名不能把两个同类型 requirement 分到两台相机。运行时还会核对每路 RGB-D/标定的 activation 和配置的 provider。

每路先配对自己的彩色与深度，再按 `maximum_skew_ms`（默认 20 ms）配对两路深度时间；缓冲有界，过旧帧丢弃，每对只消费一次。默认 `capture` 要求两路完整时钟身份（kind/source/generation）一致，不能直接比较两台设备各自的 Device 时间。可显式选择 `received`，按同一 HostMonotonic 时钟中的接收时间做 best-effort 配对；这不表示曝光同步。输出位姿、主时间戳及初始化始终以主相机为准，双路输出额外记录第二路及跨相机配对信息。

目标暂时不在某路画面时，沿用上游逐点剔除及深度匹配规则；某个深度模态没有匹配，则该次梯度/Hessian 为零，另一视角仍可约束同一个物体。沿用上游的遮挡不足回退行为，不保证遮挡时绝无误匹配，也不增加自动丢失重定位。相机断流与目标不可见不同：双路模式需要两路新帧配对，不复用旧的第二路图像，也不自动降级为单路。

`camera_in_primary` 表示固定相对安装关系。启用下述 pitch 补偿时，两台相机必须同属该刚性安装；第二路的世界位姿为主相机世界位姿乘以此外参。两台相机独立运动需要额外的动态外参接入。

双路软件回归使用两个独立的原生 provider、非 identity 外参、共享的合成采集时钟和独立输出消费者；两路图像持续发布，轮流将某一路 RGB-D 置空观测，检查另一视角仍能跟踪运动。这里的置空图像不等同于物理遮挡或目标出画的实机验收。`depth-only` 专门验证第二路深度的贡献，`baseline` 验证默认 Region+Depth，`extended` 覆盖 Texture、每视角渲染器和初始化细化：

```bash
python3 perception/m3t/scripts/verify_dual_camera.py \
  --build-dir build/m3t --composer "$RBT_PREFIX/bin/composer" \
  --output-dir target/m3t-dual-camera-verify --profile depth-only
```

输出目录须尚不存在。回归采用正式模型采样参数（`model_divides: 2`、`model_points: 200`、`model_image_size: 512`），并逐段比较主路无观测时的运动与保持上一帧估计的误差。低采样模型不保证深度单模态的稳定跟踪。

### Pitch 相机运动补偿（可选）

`camera_motion` 用原生 `camera/PitchTf` 动态 TF stream（provider 必须匹配 `pitch_provider`；AidLux 为 `camera-pitch`）描述 `world_frame <- pitch_frame`，并用相机 SDK 的 `StaticUpdates` / `StaticSnapshot` 提供同一 RGB-D activation 中的静态相机变换。`mount_in_pitch` 是 16 项 row-major 的 `T_pitch<-depth`；`arm_bases[].base_in_world` 是每个 arm 固定的 16 项 row-major `T_world<-base`。这些 arm bases 始终留在 WORLD，应用按图像更新相机相对 WORLD 的姿态；输出仍是彩色光学帧下的 `T_camera<-link`。 Pitch TF 由 activation-owned 独立任务持续接收，以免模型生成或追踪计算占用 GL 线程时阻塞反馈；本地样本缓存有界并受互斥保护。

下面只展示字段形状，identity 矩阵是格式占位值，不是可用于实机的标定。实际矩阵应根据 CAD 和当前 nominal initial estimate 填写；实机闭环验证尚未完成。`world_frame`、`pitch_frame`、`depth_frame` 和 arm 名称必须与实际 TF、初始化及模型一致。

```json
{
  "camera_motion": {
    "enabled": true,
    "world_frame": "world",
    "pitch_frame": "camera_pitch",
    "pitch_provider": "camera-pitch",
    "depth_frame": "camera_depth_optical",
    "time_basis": "received",
    "allow_unknown_uncertainty": true,
    "maximum_wait_ms": 200,
    "maximum_sample_age_ms": 1000,
    "mount_in_pitch": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
    "arm_bases": [
      {"arm":"so101_left", "base_in_world":[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]},
      {"arm":"so101_right", "base_in_world":[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]}
    ]
  }
}
```

统一 Orbbec 当前发布入口将图像 capture 时间标记为主机入口处的 `HostMonotonic` 时间；M3T 仍兼容旧 provider 的 `Device` capture 时间，不会在两种时钟间换算。RGB-D pair 要求深度和彩色图像的完整 `ClockIdentity`（kind、source、generation）一致。Pitch 样本使用接收时记录的 `HostMonotonic` 时间，当前精度未知。因此板端配置显式使用 `time_basis: "received"` 和 `allow_unknown_uncertainty: true`，这是 best-effort 关联，不表示曝光与 pitch 测量同步，也不声称精确曝光时刻。原生 pitch/mount lookup 在选中时间基准对应的时间进行；`capture` 仍是默认基准，只有图像与 pitch 时间戳具有相同完整 `ClockIdentity` 才可用。

选择策略是最近实测样本：缓冲最多 256 个样本且最多保留 10 秒；允许选中早于或晚于图像基准时间的样本，等距时选较早样本，单个样本也可用，不要求前后夹住图像时间。无 pitch feedback、时钟不匹配或样本与图像基准时间相差超过 `maximum_sample_age_ms` 时，该 RGB-D 对不会生成跟踪结果。`maximum_sample_age_ms` 默认 1000 ms；`maximum_wait_ms` 默认 200 ms，只限制启动时等待首个 pitch feedback；`maximum_interpolation_gap_ms` 默认 200 ms，保留给原生 TF lookup policy，不改变最近样本选择。输出 JSON 的 `camera_motion` 会记录 `selection_policy`、所选样本时间及有符号/绝对时间偏移。

AidLux 采集时使用：

```bash
rbt-so101-capture RUN_DIR --config ~/rbt-work/config/m3t/pitch-tracking.json
```

CAN0 恢复后，capture runner 会在同一个 Composer 中自动启动 passive pitch provider；无需另起 Composer 或下发电机命令。

### SO-101 关节状态先验

[`configs/so101_joint_prior.example.json`](configs/so101_joint_prior.example.json) 展示了双臂前四个关节的配置。M3T 直接订阅已运行驱动发布的 SO-101 `StampedBytes` 状态流，`joint_priors.service_name` 必须填写该流的实际服务名，`controller_id` 也必须与驱动一致。它不创建新的 Composer runtime、状态合成器或驱动。该协议中的关节 1–5 数值单位为度，夹爪为百分比；每项映射都必须明确给出源与模型的 arm/joint、`scale_rad_per_unit`、`offset_rad` 和弧度标准差 `standard_deviation_rad`。

示例将前四个源关节映射到官方来源模型的 `shoulder_pan`、`shoulder_lift`、`elbow_flex` 和 `wrist_flex`；源映射保持为左右驱动的同名关节，比例使用度转弧度。标准差 `0.1 rad` 只是配置示例，不是经验证的硬件校准值。示例没有映射 wrist roll 或夹爪；完成实际模型和驱动语义校准后再显式添加。视觉模型的目标 arm/joint 与硬件驱动的源 arm/joint 之间必须逐项显式配置，实现不会按名称自动推断。实现也不会假定 VLA90/VLA102 或 87° 的比例、偏置或关节约定。

AidLux 的旧数字关节 wrist mapping 配置已删除，仓库没有可复用的 `+90°/+102°` wrist 偏置。新模型的 wrist 映射、零点、base pose 和 encoder offset 都必须按实际驱动与当前模型重新审查。夹爪不映射。原有两项 `0.1 rad` 标准差是配置暂定值，不是测得的不确定度或校准结果。

`capture` 是默认且严格的时间基准：只有图像与状态样本都具有相同 capture clock 时才可关联；选取不晚于图像时间的最新样本，并受 `maximum_age_ms` 限制。缺少 capture 时间或时钟不匹配时不会生成因子，也不会自动回退到 `received`。示例显式选用 `time_basis: "received"`，它只在同一主机单调接收时钟上做有界、因果的 best-effort 关联；这不代表相机曝光时间同步。输出的 `joint_priors` 诊断会记录所用基准、样本年龄、capture 关联标志、有效先验和被拒映射。

`required` 默认是 `false`：先验缺失或所有映射都无效时仍继续图像跟踪；部分有效的映射可单独使用。设为 `true` 后，当前 RGB-D 对没有任何有效映射时会跳过该对的跟踪估计并记录诊断。映射结果超出模型 URDF 关节限位时，该项会被省略并列入 `rejected_mappings`，不会夹紧到限位；其余有效映射仍可使用。

`initialize_from_measurement` 默认启用：引擎首次建立时，将当时可用的有效先验写入 seed 关节角并据此计算 FK。之后每个 M3T optimizer 更新迭代都把测量作为标量 Gaussian 因子，与图像残差一起求解：对当前关节角 `q`、测量 `q_m` 和标准差 `σ`，向梯度项加 `(q_m - q) / σ²`，向对角 Hessian 项加 `1 / σ²`。因此每帧先验影响优化结果，但不会在每帧把 FK 直接强制设成测量角度。optimizer 支持代码通过本地构建目录生成的 overlay 接入；固定上游 checkout 和提交保持未修改。

启动与显式 `reinitialize` 的原生 Refiner 也在每次优化前根据当前关节角刷新先验残差，并在优化后投影关节、同步位姿；Refiner YAML 的迭代次数、modality 重启和可视化行为保持不变。`reinitialize(seeds, iteration, priors)` 仅使用本次传入的先验；省略 `priors` 时不沿用历史先验。

启用时，把示例的 `joint_priors` 对象合并进实际 tracking config，改成运行中驱动的真实 service/controller ID，并按实测模型逐项确认映射与不确定度。该示例的 `required: false` 和 `time_basis: "received"` 是显式选择；若省略 `time_basis`，严格默认值仍为 `capture`。

根级数值选项 `model_divides`、`model_points`、`model_image_size`、`correspondence_iterations` 和 `update_iterations` 与旧配置相同。Region/Depth 的 measured occlusion、modeled occlusion 与 silhouette/region checking 由 `occlusion` 控制；启用 modeled occlusion 或 checking 会用 OpenGL 渲染器。Depth YAML 中的 `measure_occlusions` 若存在，必须与 `occlusion.measured` 一致。

`metadata` 文件直接交给 M3T 上游对应对象的 YAML loader；这里不翻译或覆写完整参数。`configs/*_modality.example.yaml`、`optimizer.example.yaml` 和 `refiner.example.yaml` 展示了部分字段。每个 modality YAML 也可以设置上游 `visualize_*`、`display_visualization` 和保存选项；顶层 `visualization` 控制应用在 correspondence、optimization、result 阶段调用 M3T 可视化。

没有指定 Refiner YAML 时，启动细化的 `visualization_time` 默认是安全的 1 ms。指定 YAML 后保留上游值：`visualization_time: 0` 会无限等待按键，适合交互调试；正值（毫秒）可连续运行，例如设为 `1`。

约束 metadata 必须给出两个 link 各自的 joint frame 变换和六个方向标记。方向顺序是旋转 x/y/z，再平移 x/y/z。配置结构例如：

```json
"constraints": [
  {"name":"left_pair", "arm":"so101_left", "link1":"shoulder",
   "link2":"upper_arm", "type":"hard", "metadata":"hard_constraint.example.yaml"},
  {"name":"right_pair", "arm":"so101_right", "link1":"shoulder",
   "link2":"upper_arm", "type":"soft", "metadata":"soft_constraint.example.yaml"}
]
```

[`hard_constraint.example.yaml`](configs/hard_constraint.example.yaml) 与 [`soft_constraint.example.yaml`](configs/soft_constraint.example.yaml) 中的 identity frame 只说明文件格式；使用前必须按实际结构校准。约束属于单个 arm 的 optimizer，不能把左右臂或不同 optimizer 的 link 放进同一个约束。

### 显式重初始化

`--reload-on-sighup` 开启后，先用真实相机数据生成新的 measured initialization，再原子替换配置所指的初始化文件并向运行中的进程发送 SIGHUP：

```bash
rbt-m3t run --config "$M3T_DATA/config.json" \
  --reload-on-sighup
# 从新初始化器结果更新配置所指的 initialization.json 后：
kill -HUP <RBT_M3T_PID>
```

这只重新加载该 measured initialization 文件。相机序列号、坐标帧、arm 名称和 `fixed_base` 选择必须与现有 tracker 相同；模型、配置参数、metadata、模态和相机 activation 不会重载。重新初始化是显式操作，不会自动识别或恢复丢失的目标。

## 模态、GPU 与输出

Texture 使用 upstream CPU ORB（4）、BRISK（0）或 SIFT（3）。DAISY（1）和 FREAK（2）仅在 OpenCV 提供 `xfeatures2d` 时可用。CUDA ORB（5）默认关闭；只有显式以 `-DRBT_M3T_ENABLE_CUDA=ON` 构建、找到 OpenCV `cudafeatures2d` 并具备 NVIDIA CUDA 环境时才能使用。新增的 Hexagon DSP ORB 使用 `descriptor_type: 6`，须显式启用 `RBT_M3T_ENABLE_HEXAGON_ORB`；详见 [`dsp/README.md`](dsp/README.md)。CPU ORB 与 CUDA ORB 的编号保持不变。Adreno GPU 的 Zink OpenGL 路径不提供 CUDA；M3T optimizer 仍在 CPU 上运行。上游数据集 benchmark 示例未移植，不能据此宣称完整自动检测器或全 GPU optimizer 已接入。

JSON 继续保留原有采集时序、link 位姿和关节角字段，并新增实际启用的 `modalities` 数组。恰好为 `region, depth` 时 `algorithm` 仍是 `m3t_region_depth_articulated`；其他模态组合为 `m3t_articulated`。

`run` 通过 Composer 提供 `rbt.m3t.ArmTracking@1`。每条 `StampedBytes` 是一个完整 SensorLatest 结果批次，不是 Control 流或逐变换 burst。先从 capability 清单取得 stream 服务名，再查看 ABI：

```bash
rbt capability list --json
rbt stream info <capability-id-or-service>
```

安装了 RBT Python 绑定的消费者可用 `StampedBytesSubscriber` 拆开原生时序前缀和 JSON payload：

```bash
export M3T_SERVICE='从 capability 清单复制 stream 服务名'
python3 - "$M3T_SERVICE" <<'PY'
import json
import sys
from rbt.time_stream import StampedBytesSubscriber, stamped_bytes_descriptor, split_payload

subscriber = StampedBytesSubscriber.open(stamped_bytes_descriptor(sys.argv[1]))
try:
    sample = subscriber.receive(timeout=10.0)
    if sample is None:
        raise SystemExit("没有收到 M3T pose batch")
    timing, payload = split_payload(sample.payload_view())
    estimate = json.loads(bytes(payload))
    assert estimate["schema_version"] == 1 and estimate["event"] == "estimate"
    assert estimate["modalities"] == ["region", "depth"]  # 本示例的默认组合
    arms = {link["arm"] for link in estimate["links"]}
    assert arms == {"so101_left", "so101_right"}
    assert set(estimate["joint_angles_rad"]) == arms
    print({"sequence": sample.sequence, "capture": timing.capture,
           "algorithm": estimate["algorithm"], "modalities": estimate["modalities"],
           "links": len(estimate["links"]), "arms": sorted(arms)})
finally:
    subscriber.close()
PY
```

JSON `links[].pose` 是米制 `T_camera<-link` 的 16 项 row-major 矩阵；`joint_angles_rad` 按 arm/link 关节名给弧度。`frame_sequence`、`dropped_frames`、`frame_id` 与深度/彩色采集时间、配对差、`capture_clock_kind`、`capture_clock_source`、`capture_clock_generation` 和不确定度都在 JSON 内；`capture_clock_kind` 为 `HostMonotonic`（统一 Orbbec 当前发布入口）或兼容旧输入的 `Device`。StampedBytes ABI 前缀另带 capture/receive 时间。启用 Texture 时，`modalities` 会如实包含 `texture`，算法名也随模态组合变化。

## 验证状态

截至 2026-10-23，M3T 0.3.0 ARM64 DSP-enabled 安装通过 7/7 CTest，并通过 AidLux V73 CDSP 独立 ORB 测试和 extended 合成软件 profile（含 DSP 路径与 SIGHUP 重初始化）。这些结果确认 DSP 功能在设备和合成软件 fixture 上工作；相机采集、真实 SO-101 初始化与追踪仍为 NOT_RUN。DSP microbenchmark 中提取和匹配都慢于对应 CPU 路径，因此不宣称加速。验证范围、数字和证据路径见 [`VERIFICATION.md`](VERIFICATION.md) 与 [`dsp/README.md`](dsp/README.md)。2026-10-23 的 M3T 0.2.0 GPU-renderer 历史验证和 2026-10-21 开发机结果保留在验证记录中。

2026-10-24 的 SO-101 关节先验路径通过 5 项聚焦 CTest（0.71 秒）和原生软件 fixture：55 帧中 13 帧应用先验，停止发布后最后 5 帧正确报告 stale 且不保留因子；关闭先验时，独立原生消费者收到 20 帧基线结果。ARM64 CPU-only 构建也通过 5/5 CTest、隔离安装及两个 `--help` smoke checks。以上均为合成软件或构建验证；实际驱动映射、相机同步和机械臂精度仍为 NOT_RUN。完整结果见 [`VERIFICATION.md`](VERIFICATION.md)。

目标机可用新 verifier flags 复测 extended profile，输出目录应使用新路径：

```bash
python3 perception/m3t/scripts/verify_software.py \
  --build-dir build/m3t --composer "$RBT_PREFIX/bin/composer" \
  --output-dir target/m3t-verify-extended \
  --rendering hardware --profile extended --exercise-reinitialization
```

此命令使用默认 CPU Texture 后端；DSP profile 需要设备库路径和非空运行审计，按 [`dsp/README.md`](dsp/README.md) 操作。软件 fixture 只验证合成输入，不代表真实 SO-101 精度。

关节先验的原生单元和软件传输/优化验证使用单独命令；`--output-dir` 必须是尚不存在的新目录：

```bash
ctest --test-dir build/m3t --output-on-failure -R 'frame_source|model|app_config|joint_prior'
python3 perception/m3t/scripts/verify_joint_prior.py \
  --build-dir build/m3t --composer "$RBT_PREFIX/bin/composer" \
  --output-dir target/m3t-joint-prior-verify-rerun-1
```

软件 verifier 使用原生 RBT RGB-D 与关节状态流及 native optimizer；它检查有效先验的应用、停止状态发布后的 stale 因子剔除和子进程清理。它是合成软件 fixture，不测真实相机曝光同步、实物模型映射或硬件跟踪精度。2026-10-24 的结果与 ARM64 CPU-only 验证记录见 [`VERIFICATION.md`](VERIFICATION.md)。

## Fixed-camera CAD mount refinement

The opt-in `mount_refinement` configuration estimates each mount against a fixed
prior anchor. Combined with `encoder_reference_links`, it keeps both the base
pose and visual joint corrections free, with separate soft priors. Used alone,
it holds revolute joints at complete live encoder measurements.
Optional `visual_link_weights` scale visual gradients and Hessians per mesh link.
See [configuration, selected experiment and acceptance limits](docs/cad-mount-refinement.md).
This implementation is not a physical tracking-accuracy acceptance.

## 编码器参考 link 与关节视觉修正

在原生 tracking config 中设 `"encoder_reference_links": true`，并为每只臂的
全部旋转关节配置 `joint_priors`，启用 `required` 和 `initialize_from_measurement`。
每帧由编码器更新各关节的虚拟参考 link。物理 link 在参考上保留可优化的关节轴，
由 M3T 的旋转软约束限制视觉修正幅度。关节的 `standard_deviation_rad`
决定该软约束的权重。新一帧带来新的编码器参考，上一帧的局部视觉修正继续作为初值。

估计报文中的 `joint_priors.priors` 是输入，`joint_angles_rad` 是视觉优化后的角度。
这一模式只使用完整且有效的反馈，缺失时跳过该帧。可以同时启用
`mount_refinement.enabled=true`，并将初始化中每臂的 `fixed_base` 设为 `false`：
底座的六自由度位姿和关节角共同参与视觉优化，各自保留软先验。底座先验中心保持为
初始化位姿，不随每帧估计漂移；标准差决定约束权重，并非误差硬上限。
组合输出的 `mount_refinement.joints_locked_to_feedback=false`。
官方原理见
[作者论文 §6.2.4](https://elib.dlr.de/203345/1/1696259.pdf)。

## Native RBT TF publication

Pass `--publish-tf` to native `run`. The tracking component advertises
`perception/M3tTf` and publishes complete
`<arm>/cad_mount → <arm>/base_link → <arm>/<link>` trees from each fresh estimate
using its capture clock. Encoder reference links support either fixed tracking
bases or soft base priors via `mount_refinement`. Set `tf_nominal_initialization` to preserve
separate nominal CAD frames. The dashboard reads the service's preview via
`--m3t-preview-packet`.
Use `rbt-m3t-tf-query` to discover, subscribe and query these native transforms.
The existing JSON/HTTP views remain diagnostics, rather than the native TF transport.
See [frame names, timing and consumption](docs/native-tf-output.md).
