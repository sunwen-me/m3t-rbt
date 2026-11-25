# Gripper pose output

The dashboard endpoint GET /api/overlay/gripper-poses returns the latest valid M3T pose associated with an image. It is a capture-rate preview API, not a control stream.

The response schema is rbt-m3t-gripper-poses-v1. coordinate_frame identifies the camera frame. Each arm entry includes:

| Field | Meaning |
|---|---|
| T_camera_from_base_link | Estimated arm base pose in the camera frame |
| T_base_link_from_gripper_link | FK from the same estimate's joint angles and runtime model |
| T_camera_from_gripper_link_fk | Product of the preceding transforms |
| T_camera_from_gripper_link_m3t | Direct gripper-link pose from that estimate |
| max_matrix_abs_difference | Maximum element-wise difference between FK and direct output |

Transforms are row-major 4×4 matrices. Translation is in metres at indices 3, 7, and 11. T_A_from_B maps coordinates in B into A. The page displays translation in millimetres.

The FK uses joint angles from the same M3T estimate, converted from radians for the model FK function. The model FK and native result must agree within 2e-4 per matrix element.

The response includes frame identity, color capture timestamp, capture clock identity, preview age, and joint-prior diagnostics. If tracking is disabled, stale, or has an error, the endpoint returns HTTP 503. Its sample freshness limit is two seconds; consumers should apply their own task-specific age and tracking-quality limits.

The target is gripper_link, the model frame rather than a calibrated physical contact point. Native TF publication is described in [native-tf-output.md](native-tf-output.md).
