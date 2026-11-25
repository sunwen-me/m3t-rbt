#pragma once

#include <rbt_m3t/camera_motion.hpp>
#include <rbt_m3t/model.hpp>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace rbt_m3t {

struct SecondaryCameraConfig {
    std::string provider;
    std::string serial;
    std::string coordinate_frame;
    // Fixed T_primary_color_optical<-secondary_color_optical, metres.
    Eigen::Affine3f camera_in_primary { Eigen::Affine3f::Identity() };
    std::uint64_t maximum_skew_ms { 20U };
    std::string time_basis { "capture" };
};

struct TrackingConfig {
    RobotModel model;
    std::vector<ArmSeed> arms;
    EngineOptions engine;
    std::string coordinate_frame;
    std::string serial;
    std::filesystem::path initialization_path;
    JointPriorConfig joint_prior;
    CameraMotionConfig camera_motion;
    std::optional<SecondaryCameraConfig> secondary_camera;
    std::vector<ArmSeed> tf_nominal_arms {};
};

struct TrackingInitialization {
    std::vector<ArmSeed> arms;
    std::string coordinate_frame;
    std::string serial;
};

[[nodiscard]] TrackingInitialization load_tracking_initialization(
    const std::filesystem::path& path, const RobotModel& model);

// Config: schema_version=1, robot_model path, initialization path,
// cache_directory path, optional EngineOptions numeric keys.
// Initialization is separate, measured data produced by rbt-m3t-initialize.
[[nodiscard]] TrackingConfig load_tracking_config(const std::filesystem::path& path);

// UTF-8 application JSON, schema_version=1; native StampedBytes carries timing.
// Matrices are T_camera<-link, row-major, translations metres; joint values radians.
[[nodiscard]] std::string tracking_json(const TrackingEstimate& estimate,
                                       const RectifiedPair& pair,
                                       std::uint64_t frame_sequence,
                                       std::uint64_t dropped_frames,
                                       const CameraMotionSelection* motion = nullptr,
                                       const RectifiedPair* secondary_pair = nullptr,
                                       const SecondaryCameraConfig* secondary_config = nullptr);

struct TrackingCapability {
    static constexpr char TYPE_NAME[] = "rbt.m3t.ArmTracking@1";
};

} // namespace rbt_m3t
