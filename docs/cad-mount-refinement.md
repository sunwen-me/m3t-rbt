# CAD mount refinement

CAD mount refinement estimates a small six-degree-of-freedom correction to a fixed camera-to-base reference. Encoder reference links can be enabled with it so the visual optimizer also adjusts joints under encoder soft priors.

The objective combines image residuals, a CAD mount prior, and optional joint priors:

    T_camera_base = T_camera_CAD_base * Delta
    q_joint = q_encoder + delta_q_visual

    cost = visual_cost
         + translation(Delta)^2 / sigma_translation^2
         + rotation_log(Delta)^2 / sigma_rotation^2
         + sum(delta_q_visual^2 / sigma_joint^2)

The CAD anchor is fixed during tracking. The base correction is optimized around that anchor, with default translation sigma 0.02 m and rotation sigma 5 degrees. These sigmas set the soft-prior strength; they are not hard correction limits. In combined mode, each joint has an encoder reference link and a soft constraint using that joint's configured standard deviation. Base and joints are solved by the same optimizer.

## Configuration

Enable both features in a tracking configuration:

    "encoder_reference_links": true,
    "mount_refinement": {
      "enabled": true,
      "translation_standard_deviation_m": 0.02,
      "rotation_standard_deviation_rad": 0.08726646259971647
    }

Mount refinement requires fixed_base=false for each arm. Mount refinement and encoder reference links both require enabled, required joint_priors with initialize_from_measurement=true. Mount refinement uses a fixed camera and cannot be combined with camera_motion; either mode disables StaticDetector and initialization Refiner.

With mount refinement alone, revolute joints remain locked to encoder feedback. With encoder reference links alone, arm bases remain fixed and joints are visually optimized under their priors. With both enabled, bases and joints are optimized together. Without either option, existing tracking behavior is unchanged.

Output reports the nominal and refined base poses, correction magnitudes, and whether joints were locked to feedback. Internal CAD anchor and encoder reference nodes are not published as robot links. Reinitialization resets the current correction to the requested nominal CAD mount.

## Per-link visual weights

The optional visual_link_weights object scales enabled visual modalities for each named mesh link. Unspecified links use 1.0; valid weights are 0.01–100. For example:

    "visual_link_weights": {
      "gripper_link": 5.0,
      "moving_jaw_so101_v1_link": 5.0
    }
