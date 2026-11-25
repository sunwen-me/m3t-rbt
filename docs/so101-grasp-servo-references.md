# SO101 抓取伺服实现调研（2026-11-16）

目标：复用现有 GraspGenX、固定 −61° 双相机与 M3T，完成普通 SO101 的桌面抓取。用户明确本轮不需要避障规划。本记录是源码调研和实现建议，不是新控制器部署、实机运动或精度验收。

## 核对方法

直接阅读官方驱动、运动学代码与社区抓取循环。区分电机位置闭环、视觉反馈更新、命令插值和任务状态机；没有把命令发送频率当成视觉频率，也没有把 IK 数值残差当成夹爪实际精度。

### 官方 LeRobot：位置接口和 IK

- [SO follower 驱动](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)：SO101 使用同一 follower 实现；读取 `Present_Position`，把关节位置目标写入 `Goal_Position`。可配置相对位置目标幅度限制。身体关节支持度数或归一化模式，夹爪有独立归一化模式。
- [运动学实现](https://github.com/huggingface/lerobot/blob/main/src/lerobot/model/kinematics.py)：`RobotKinematics.inverse_kinematics()` 用当前关节值作为初值，将末端目标转换为关节位置；位置、姿态有独立权重，返回值为角度。
- [官方末端动作表示说明](https://huggingface.co/docs/lerobot/v0.6.0/action_representations)：给出末端增量、参考系、IK 和电机目标之间的处理关系。

可借鉴位置接口及增量 IK 的结构；不要求将现有 RBT 栈换成 LeRobot。示例的归一化数值、轴向、增益不能直接当成我们 v2 的角度命令。IK 返回结果后仍需检查 FK 的目标残差。

### mmporong/so101-mobile-manipulation：局部图像雅可比，移动后再观察

固定阅读版本：`eebd5c0723d9b596ad4e6bcbc2c4d7e9b3aecf17`。

- [pick_wrist.py](https://github.com/mmporong/so101-mobile-manipulation/blob/eebd5c0723d9b596ad4e6bcbc2c4d7e9b3aecf17/pick_wrist.py)：先打开夹爪、到示教观察高度，再根据目标像素误差和标定的图像雅可比计算平面小步修正；每次移动完成、等待后重新观察。随后对齐、下降、闭合、抬起检查。
- [wrist_calib.py](https://github.com/mmporong/so101-mobile-manipulation/blob/eebd5c0723d9b596ad4e6bcbc2c4d7e9b3aecf17/wrist_calib.py)：腕部 RGB 中的颜色区域、质心和角度提供视觉特征。
- [pick_demo.py](https://github.com/mmporong/so101-mobile-manipulation/blob/eebd5c0723d9b596ad4e6bcbc2c4d7e9b3aecf17/pick_demo.py)：HTTP IK 请求由机械臂服务转换并插值为关节位置目标。

可借鉴反馈—小步运动—重新测量、步长上限和误差不下降时退出。它是局部 move–observe，不是任意起点的连续六自由度位姿控制；源码阈值不等于实物精度。

### ultrafro/garbagecollector：新图像修正方向，连续发送小步关节目标

固定阅读版本：`b92487fb2dd58028e19306f8626f3edf65f74dec`。

- [control_server.py](https://github.com/ultrafro/garbagecollector/blob/b92487fb2dd58028e19306f8626f3edf65f74dec/autonomy/control_server.py)：从当前测得且已视觉瞄准的姿态开始接近。每当有新的目标图像，按中心误差修正接近方向；运动端用小步 IK 和插值继续发送关节目标。新图像、关节反馈和命令发送各有自己的时序。
- [pi/lekiwi_ws.py](https://github.com/ultrafro/garbagecollector/blob/b92487fb2dd58028e19306f8626f3edf65f74dec/pi/lekiwi_ws.py)：接收关节位置消息，调用电机位置写入接口。

可借鉴低频视觉修正与较快的位置目标更新分离。它使用腕部目标框和接近射线，不是 GraspGenX 六自由度抓取姿态伺服；源码还包含姿态约束求解失败时退回仅位置求解的分支。目标暂时不可见时保留方向的策略也不直接照搬到我们系统。未建立独立的物理抓取精度结论。

### XWen1024/LeRobot_SO101_Visual_Servo：预设位姿加像素接近状态机

- [servo_controller.py](https://github.com/XWen1024/LeRobot_SO101_Visual_Servo/blob/main/src/servo_controller.py)：支持先移动到录制的预抓取姿态，图像更新时跟踪目标；在接近状态依据像素中心误差调整关节，以目标框面积作为接近/闭合条件。

可借鉴状态机和简单关节位置执行。它不能作为普通 SO101 已完成通用三维精准抓取或任意起点全程视觉伺服的证据。

## 对我们现有系统的建议

1. GraspGenX 选定本轮候选，保留输入点云参考帧、时间戳和夹爪坐标定义；不要每个电机周期重新生成候选。
2. 核对旧 GGX 软鳍资产与当前机械臂的末端坐标关系。用本轮有效 M3T TF 把候选换算到所选机械臂底座，并验证候选的 IK 残差与关节限位。
3. 从初始位姿就启用小步位置闭环：当前反馈 → 当前末端估计 → 目标位姿误差 → 有界小步末端目标 → 以实测关节为初值的 IK → 关节位置命令。
4. 预抓取、最终接近、闭合和抬起分别作为状态目标。视觉测量有效时从开局参与修正；预抓取位置是中间目标，不能据此推断之前没有伺服。
5. 分开处理视觉新帧与关节控制周期。视觉过期、机械臂反馈失效、无合格 IK 或误差不收敛时停止继续推进目标；具体频率和步长需基于本机实际反馈时延设定。
6. 当前 M3T 为底座软修正、关节固定编码器；末端输出含 FK，尚不是独立测量所有末端机械误差。先接估计位姿闭环，再检查视觉/模型收敛与实物对准是否一致。

以上是基于源码的适配建议，尚未新增或运行机器人控制循环。此次未改电机参数、发送关节目标或启停部署服务。

## 证据保存

本地源码快照与 SHA-256 清单：`target/so101-servo-references-20261116/`，按仓库规则不加入 Git。社区两项引用固定到上述版本；官方与 XWen 链接为读取时的 main，快照记录具体文件哈希。现有部署边界见 [GGX 核查](../../../deploy/20261116-graspgenx-existing-deployment-audit.json) 和 [M3T TF 接口](native-tf-output.md)。
