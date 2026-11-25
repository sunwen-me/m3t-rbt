# RBT M3T

RBT M3T integrates the DLR-RM M3T tracker with the RBT C++ SDK. It tracks robot links from RGB-D input and an initialization file. Region and Depth are enabled by default; Texture is optional.

The tracker refines the supplied initialization. It does not search the full scene or relocalize a lost target.

## Build

Requires RBT C++ SDK 0.8, C++20, Eigen, OpenCV 4, GLFW, GLEW, OpenGL, OpenMP, and the RBT Orbbec driver source package. A usable OpenGL context is required even when preview is disabled.

Set the SDK, pinned upstream checkout, and Orbbec source paths:

    export RBT_PREFIX=/path/to/rbt-sdk
    export RBT_M3T_UPSTREAM_ROOT="$PWD/target/m3t-upstream"
    ./scripts/bootstrap_native.sh

    cmake -S . -B target/build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=ON \
      -DCMAKE_PREFIX_PATH="$RBT_PREFIX" \
      -DRBT_M3T_UPSTREAM_ROOT="$RBT_M3T_UPSTREAM_ROOT" \
      -DRBT_M3T_ORBBEC_SOURCE_DIR=/path/to/rbt_robot_ws/drivers/orbbec
    cmake --build target/build --parallel 4
    ctest --test-dir target/build --output-on-failure

The upstream M3T revision and source overlays are recorded in [UPSTREAM.toml](UPSTREAM.toml). The Orbbec driver package is provided by the surrounding RBT workspace.

## Model and run

The SO-101 model recipe is in [models/so101_lerobot_soft_fin](models/so101_lerobot_soft_fin/README.md). Generate the URDF and M3T robot model with:

    mkdir -p target/so101
    python3 models/so101_lerobot_soft_fin/build_model.py \
      --output target/so101/model
    python3 scripts/prepare_robot.py \
      --urdf target/so101/model/so101_lerobot_soft_fin.urdf \
      --output-dir target/so101/robot

Configuration paths are relative to the configuration file; update the example paths when copying a config. With Composer and the aligned camera running, capture a fresh snapshot and initialize from the measured joint angles:

    rbt-m3t probe --provider orbbec335 --snapshot target/so101/snapshot
    rbt-m3t-initialize \
      --model target/so101/robot/robot.json \
      --snapshot target/so101/snapshot \
      --output target/so101/initialization.json \
      --joint-angles target/so101/joint_angles.json
    rbt-m3t run --config /path/to/tracking.json

The camera driver must publish depth aligned to color. Capture-time association requires matching clock identities; received-time association is best-effort host timing and does not imply exposure synchronization.

## References

- [Configuration examples](configs/)
- [Native TF output](docs/native-tf-output.md)
- [Gripper pose output](docs/gripper-pose-output.md)
- [CAD mount refinement](docs/cad-mount-refinement.md)
- [Optional Hexagon DSP ORB](dsp/README.md)

Build outputs, snapshots, and calibration data belong outside tracked source files.
