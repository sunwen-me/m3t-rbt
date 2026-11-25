# 固定相机下的 CAD 安装修正实验

本模式用于固定相机和 v2=URDF 零点条件。CAD 安装变换作为底座软先验的名义参考，
每臂底座保留六自由度安装修正。与 `encoder_reference_links=true` 组合时，
每个关节也保留视觉优化自由度，以当前只读编码器反馈为软先验。它不会向机械臂发送指令。

## 状态与代价

```text
T_camera_base = T_camera_CAD_base * Delta
q_joint = q_current_encoder + delta_q_visual

cost = sum_link(weight_link * visual_region_depth_cost_link)
     + ||translation(Delta)||^2 / sigma_translation^2
     + ||rotation_log(Delta)||^2 / sigma_rotation^2
     + sum_joint(delta_q_visual^2 / sigma_joint^2)
```

每臂的 optimizer 使用无 Body 的固定 CAD anchor 为根，实际 base 为其六自由度子节点。
原生 M3T SoftConstraint 在 anchor 与 base 之间约束全部六轴，
零死区，默认平移标准差 0.02 m、旋转标准差 5°。这是围绕固定 CAD 参考的惩罚，
不是单次优化增量阻尼，也不会每帧将参考更新成视觉估计。

相机保持固定。组合模式在每个 revolute joint 前保留无 Body 的编码器参考节点，
每帧在计算视觉对应关系前更新参考节点并保留已有视觉关节修正。
参考节点与物理节点之间的旋转 Z 轴 SoftConstraint 使用该关节反馈的标准差，
底座与关节由同一 optimizer 联合求解，互不替换对方的先验。缺失、重复或越界的关节输入不能继续估计。
外部反馈仍保留各臂 v2 文件 hash、只读身份和接收时间检查；接收时钟不代表曝光同步。

仅开启 `mount_refinement` 时仍沿用原有锁关节实验：所有 revolute 后代为零自由度，
`q_joint = q_current_encoder`，视觉只估计底座安装修正。组合模式显式输出
`encoder_reference_links=true` 和 `mount_refinement.joints_locked_to_feedback=false`；
仅开启 mount refinement 时后一个标志为 `true`。

显式 reinitialize 将 CAD anchor 更新为请求的名义安装，清空旧 Delta，并重置反馈关节。
输出只包含原模型 link 身份，内部 CAD anchor 与编码器参考节点不作为物理 link 发布。
每臂输出名义/修正后 base、平移与旋转修正量，便于检查安装修正大小和变化。

## 配置

在具有完整、required、initialize_from_measurement 的 joint_priors 配置中增加：

```json
"encoder_reference_links": true,
"mount_refinement": {
  "enabled": true,
  "translation_standard_deviation_m": 0.02,
  "rotation_standard_deviation_rad": 0.08726646259971647
}
```

初始化的每臂必须 `fixed_base=false`。该实验不与动态 camera_motion、StaticDetector 或
初始化 Refiner 混用。`encoder_reference_links` 所需的完整关节软先验仍必须启用。
未开启 mount refinement 时，现有跟踪行为保持。

网页增加 `--m3t-mount-refinement`，只在显式 CAD nominal prior、固定 pitch、
实时反馈缓存和 native prior service 齐备时允许。实验入口与旧版本分开保存，
不会将本轮估计自动写回永久 TF 或 CAD prior。

夹爪优先的视觉目标可作为独立变量，不把“先验收到”当作末端精度验收。

## 末端刚性结构的视觉优先

可增加 `--m3t-end-effector-priority`，要求同时启用 mount refinement。
它生成以下配置，未列出的 mesh link 权重仍为 1：

```json
"visual_link_weights": {
  "gripper_link": 5.0,
  "moving_jaw_so101_v1_link": 5.0
}
```

权重同时乘到各视角 Region/Depth/Texture 的梯度与 Hessian，CAD 安装先验不缩放。
所有 Body 与遮挡几何保留。原生输出回报实际配置，网页要求与请求一致。
这是一个可独立对照的选项；本次 5 倍实验出现过大的底座旋转且末端仍有错位，
未选为当前默认。软底座与软关节组合无需改变末端权重，未配置时仍为默认 1。

## 数值稳定性与本次边界

M3T 的 float 位姿增量连续复合会积累非正交误差。
安装修正每轮优化后对旋转 quaternion 归一化，再投影自由关节并传播完整机械链，保留平移。
早期锁关节模式的实机连续采样检查过关节输出与 applied prior 完全一致，
旋转正交及行列式误差小于 1e-6；这不代表软关节组合模式的实机验收。

锁关节实验曾验证其约束结构、反馈传递和原生/网页诊断可运行。
软关节组合的本地测试覆盖关节可偏离编码器、反馈标准差继续生效及 CAD 名义参考重初始化。
这些测试不代表实机视觉匹配或物理安装修正已验收，也未写回永久 TF。
当前左右臂共享一套 soft-fin mesh，仍需确认实物末端装配是否确实相同。
