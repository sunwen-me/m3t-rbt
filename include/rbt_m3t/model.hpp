#pragma once

#include <rbt_m3t/frame_source.hpp>
#include <rbt_m3t/joint_prior_types.hpp>
#include <Eigen/Geometry>

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rbt_m3t {

struct ModelLink {
    std::string name;
    std::filesystem::path mesh; // Empty for a purely kinematic / virtual frame.
};

struct ModelJoint {
    std::string name, type, parent, child;
    Eigen::Affine3f origin { Eigen::Affine3f::Identity() }; // T_parent<-child at q=0.
    Eigen::Vector3f axis { Eigen::Vector3f::UnitZ() };
    double lower { 0.0 }, upper { 0.0 };
};

struct RobotModel {
    std::string name, root_link;
    std::vector<ModelLink> links;
    std::vector<ModelJoint> joints;
};

struct ArmSeed {
    std::string name; // Distinct frame prefix, e.g. so101_left / so101_right.
    Eigen::Affine3f base_in_camera { Eigen::Affine3f::Identity() };
    std::map<std::string, double> joint_angles;
    bool fixed_base { false };
};

// Extra constraints belong to one arm's optimizer. Metadata uses upstream
// Constraint/SoftConstraint YAML, including joint frames and constrained axes.
struct ConstraintOptions {
    std::string name, arm, link1, link2;
    std::filesystem::path metadata;
    bool soft { false };
};

// Fixed-camera CAD mount refinement: estimate a six-DoF mount correction per
// arm with a soft prior around its seed. Joints follow fresh encoder readings;
// encoder_reference_links also leaves their visual corrections free.
struct MountRefinementOptions {
    bool enabled { false };
    double translation_standard_deviation_m { 0.02 };
    double rotation_standard_deviation_rad { 0.08726646259971647 };
};

struct EngineOptions {
    std::filesystem::path cache_directory;
    int model_divides { 2 }, model_points { 200 }, model_image_size { 512 };
    int correspondence_iterations { 3 }, update_iterations { 2 };
    bool use_region { true }, use_depth { true }, use_texture { false };
    bool measured_occlusions { true }, modeled_occlusions { false }, region_checking { false };
    int occlusion_image_size { 512 };
    // Upstream metadata supplies the complete per-object tuning interface.
    std::filesystem::path region_metadata, depth_metadata, texture_metadata;
    std::filesystem::path optimizer_metadata, refiner_metadata;
    bool use_static_detector { false }, refine_initialization { false };
    bool visualize_correspondences { false }, visualize_optimization { false }, visualize_results { false };
    std::vector<ConstraintOptions> constraints;
    MountRefinementOptions mount_refinement;
    // Relative visual residual weights; unspecified rigid links use 1.0.
    std::map<std::string, double> visual_link_weights;
    // Track visual corrections relative to a complete set of encoder readings.
    bool encoder_reference_links { false };
};

struct LinkEstimate {
    std::string arm, link;
    Eigen::Affine3f in_camera { Eigen::Affine3f::Identity() };
};

struct MountRefinementEstimate {
    std::string arm;
    Eigen::Affine3f nominal_base_in_camera { Eigen::Affine3f::Identity() };
    Eigen::Affine3f refined_base_in_camera { Eigen::Affine3f::Identity() };
    double translation_correction_m { 0.0 };
    double rotation_correction_rad { 0.0 };
};

struct TrackingEstimate {
    std::vector<LinkEstimate> links;
    std::map<std::string, std::map<std::string, double>> joint_angles;
    std::vector<std::string> modalities { "region", "depth" };
    JointPriorReport joint_prior;
    bool joints_locked_to_feedback { false };
    std::vector<MountRefinementEstimate> mount_refinement;
    std::map<std::string, double> visual_link_weights;
    bool encoder_reference_links { false };
};

[[nodiscard]] RobotModel load_robot_model(const std::filesystem::path& path);
[[nodiscard]] std::vector<LinkEstimate> forward_kinematics(const RobotModel& model,
                                                          const ArmSeed& seed);
[[nodiscard]] std::vector<ArmSeed> seeds_with_joint_priors(
    const RobotModel& model, const std::vector<ArmSeed>& seeds,
    const std::vector<JointAnglePrior>& priors);

// All OpenGL/M3T creation, calls and destruction belong to one tracking thread.
class TrackerEngine {
  public:
    // Both image sets use camera_in_world. Seeds and outputs use the primary
    // camera frame; each view contributes modalities to the same body links.
    TrackerEngine(RobotModel model, std::vector<ArmSeed> seeds, EngineOptions options,
                  std::shared_ptr<CameraImages> images,
                  std::shared_ptr<CameraImages> secondary_images = {});
    ~TrackerEngine();
    TrackerEngine(const TrackerEngine&) = delete;
    TrackerEngine& operator=(const TrackerEngine&) = delete;
    void set_up();
    // Explicit reset from caller-supplied seeds; no automatic pose detection.
    // Arm identities and fixed-base choices must match the constructed engine.
    // Encoder reference mode requires fresh priors for every revolute joint
    // on each step and reinitialization.
    void reinitialize(const std::vector<ArmSeed>& seeds, int iteration = 0,
                      const std::vector<JointAnglePrior>& priors = {});
    [[nodiscard]] TrackingEstimate step(
        int iteration, const std::vector<JointAnglePrior>& priors = {});
  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rbt_m3t
