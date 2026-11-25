# Native M3T TF output

Run with --publish-tf to publish estimated transforms through the RBT capability perception/M3tTf. The publisher runs in the tracking component activation.

## Frames and modes

The default frame prefix is empty. For each arm, the path is:

    orbbec335/color_optical
      -> so101_left/cad_mount
      -> so101_left/base_link
      -> model links through gripper_link

The right-arm path substitutes so101_right. The robot's existing left_base and right_base static references remain unchanged.

With a fixed camera, the publisher supports CAD mount refinement with encoder-locked joints, encoder reference links with fixed bases and visually refined joints, or both together. In combined mode, the base is refined under the CAD prior and joints remain visually optimized under encoder soft priors. The output reports mount_refinement.joints_locked_to_feedback=false.

In encoder reference mode, tf_nominal_initialization supplies the CAD anchor separately from initialization, which supplies the tracking seed. Their camera and arm identities must match. For each capture, let T_C_nominal be the camera-from-CAD-mount pose and T_C_estimated the estimated camera-from-base pose:

    T_cad_mount_from_base = inverse(T_C_nominal) * T_C_estimated
    T_camera_from_base = T_C_nominal * T_cad_mount_from_base

Each model-link edge is inverse(T_camera_parent) * T_camera_child. Translation is in metres; quaternions use XYZW order. gripper_link is a model link, not a calibrated contact TCP.

## Timing and transport

Every edge in a published path carries the RGB-D depth capture timestamp and clock identity. The nominal camera-to-CAD-mount edge is emitted for each valid capture so the path is self-contained at that timestamp. Capture uncertainty is preserved.

A missing or incompatible clock identity, a future timestamp, or an estimate older than 500 ms suppresses publication. Inference completion time is used only for freshness checks; it does not restamp an estimate. The configured joint-feedback gate must also pass.

Transform messages are sent one edge at a time, not as an atomic tree batch. The sender retries queue-full records for up to 25 ms per batch, checking freshness before retry. Consumers must query a complete path at one capture timestamp. transform_epoch remains zero during an activation; a new activation receives a new RBT capability authority.

## Query

The query tool resolves the capability through Composer and checks the provider identity:

    rbt-m3t-tf-query \
      --provider TRACKING_COMPONENT_ID \
      --parent orbbec335/color_optical \
      --child so101_left/gripper_link \
      --count 3 --timeout-sec 10

Subscribe to perception/M3tTf and ingest its records into an RBT TransformBuffer. Subscribe to the robot's static and pitch TF sources as needed to resolve paths from upstream robot frames. Advertising the capability does not subscribe existing consumers.
