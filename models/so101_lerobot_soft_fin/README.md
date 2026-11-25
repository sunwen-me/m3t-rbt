# SO-101 LeRobot soft-fin model

This recipe builds the SO-101 follower model used by this M3T package. The official LeRobot URDF and asset revision are pinned in upstream.lock.json, with file hashes recorded there. Project geometry adds the palm, moving jaw, soft fins, and #34 bracket.

The active joints retain the official names shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, and gripper, with their official origins and axes. The model widens shoulder_lift and wrist_flex to ±110°, elbow_flex to [-1.69 rad, 100°], and wrist_roll to ±180° for the configured SO-101 travel. The generated model records these overrides in model_provenance.json. The official gripper_frame_link remains a nominal frame; soft-fin contact geometry is not calibrated.

Build the URDF and M3T model files from the repository root:

    mkdir -p target/so101
    python3 models/so101_lerobot_soft_fin/build_model.py \
      --output target/so101/model
    python3 scripts/prepare_robot.py \
      --urdf target/so101/model/so101_lerobot_soft_fin.urdf \
      --output-dir target/so101/robot

prepare_robot.py requires a new output directory. Generate initialization separately from an RGB-D snapshot and the measured joint angles for that snapshot; the model build does not create one. The config examples use paths relative to the configuration file.

Run the model builder tests with:

    python3 -m unittest discover \
      -s models/so101_lerobot_soft_fin -p 'test_build_model.py' -v

The upstream URDF and assets are listed in [upstream.lock.json](upstream.lock.json). The source URDF is Apache-2.0; see [licenses/LICENSE](licenses/LICENSE). Custom geometry provenance is in [soft_fin.lock.json](soft_fin.lock.json).
