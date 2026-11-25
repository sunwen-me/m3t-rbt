# Calibration tools

This directory contains offline calibration utilities, read-only feedback and preview tools, and their tests.

- handeye.py fits camera-to-arm and gripper-to-marker transforms from a manifest of fixed-camera observations.
- joint_angle_audit.py checks saved encoder, model, initialization, and tracking data.
- gripper_calibration.py maps gripper feedback through the configured encoder endpoints.
- model_overlay.py and overlay_preview.py project model links and compare them with camera frames.
- shared_board.py and wrist_intrinsics.py handle ChArUco observations and wrist-camera intrinsics.
- aruco_web.py, shared_board_web.py, and m3t_preview.py provide camera and feedback preview interfaces.

Use each script's --help for its inputs and output format. The preview tools consume existing camera and feedback streams; they do not issue motor goals. Save sessions and generated results outside the source tree.

Focused tests:

    python3 -m unittest discover -s calibration -p 'test_handeye.py'
    python3 -m unittest discover -s calibration -p 'test_model_overlay.py'
    python3 -m unittest discover -s calibration -p 'test_shared_board.py'
