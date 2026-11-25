# SO101 v2 到模型的角度核查（2026-11-15）

用户明确确认：**当前 v2 标定的 0 点就是当前 SO101 URDF 的 0 点**。
这是本次采用的已知安装/标定条件，不再把零点视作待用户补充的测量。
本次没有重新实测关节零位或正方向。

## 使用的关系

前五轴按照各臂自己的 v2 文件换算：

```text
midpoint_tick = (range_min + range_max) / 2
v2_degree = (Present_Position - midpoint_tick) * 360 / 4095
q_model_rad = v2_degree * pi / 180
```

`Present_Position` 是驱动读取到的电机位置；消费端不再次减 `homing_offset`。
两臂继续使用相同关节正号，不根据左右安装位置翻号，也不从单帧视觉结果反推偏置。
夹爪是独立的百分比输入，沿用各臂 v2 端点行程映射，不应用前五轴公式。

已读取 AidLux 实际 `runtime/so101/so101_rbt/{calibration,backend,service}.py`：
驱动的前五轴输出为上述 degree，服务直接发布这份输出，没有额外 policy-frame 偏置。
本机 LeRobot `ba83e4cf43737e2665be0aafcb5f513b83c7d0f3` 的
`motors_bus.py` degree 分支和 `model/kinematics.py` 使用相同约定。
当前官方源 URDF 固定于 `lerobot/robot-urdfs`
`04d113b3a280b5d27264dbbf56f9a98c94b68321`。

## 对正在运行的双臂实例的核查

读取当前反馈缓存、两份 v2 文件、canonical/runtime 模型、实际
`tracking.json`、`initialization.json` 和原生输出的 12 项关节先验。
初始化与其自身 provenance 中的反馈比较，避免拿当前姿态与历史初始化混比。
实时原生先验与其图像前后读取的反馈比较。

实际 12 项先验均收到，左右来源完整，年龄 32.92908 ms（接收时钟）；
数值都在图像前后读取的预期区间内。
这确认了已捕获实例中的软件换算和先验传递，不是跟踪贴合精度验收。

右肘本次原始值 3074 tick，v2 中点 1962 tick，对应
97.758241758°。它超过 v2 记录上端点 3065 tick 共 9 tick；
超过 canonical elbow 上限 1.69 rad（96.829867377°）共 0.928374381°。
这两个边界差异分别报告，不据此自动加减关节偏置。
已有运行模型的 ±100° 容许范围只解决启动范围检查，不是一次零位修正。

## 可重复核查

`calibration/joint_angle_audit.py` 仅读取保存文件，不连接串口。
用 `--v2-zero-is-model-zero` 显式记录用户确认的零点条件。
这里的 `--robot-model` 应给 canonical 模型，避免运行副本较宽的范围隐藏差异。

```bash
python3 perception/m3t/calibration/joint_angle_audit.py \
  --feedback target/so101-angle-mapping-20261115/feedback.json \
  --left-calibration target/so101-angle-mapping-20261115/left-calibration.json \
  --right-calibration target/so101-angle-mapping-20261115/right-calibration.json \
  --robot-model target/so101-angle-mapping-20261115/robot.canonical.json \
  --tracking-config target/so101-angle-mapping-20261115/tracking.json \
  --initialization target/so101-angle-mapping-20261115/initialization.json \
  --v2-zero-is-model-zero \
  --output target/so101-angle-mapping-20261115/angle-audit.json
```

原始证据保存在开发机 `target/so101-angle-mapping-20261115/`，不纳入 Git。
本次没有切换实机跟踪进程、修改驱动换算、标定文件或电机寄存器。
