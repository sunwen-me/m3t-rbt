#include <rbt_m3t/app_config.hpp>
#include <rbt_m3t/joint_prior.hpp>

#include <rbt/tf_math.hpp>
#include <rbt/transforms.hpp>
#include <rbt_orbbec/driver.hpp>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace rbt_m3t {
namespace {

constexpr float kRigidTolerance = 1.0e-4F;
constexpr std::size_t kMaximumPoseLinks = 64U;
constexpr std::size_t kMaximumConstraints = 64U;
constexpr std::size_t kMaximumNameBytes = 256U;

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument(message);
}

[[nodiscard]] bool is_number(const cv::FileNode& node) noexcept {
    return node.isInt() || node.isReal();
}

[[nodiscard]] cv::FileNode required(const cv::FileNode& object,
                                    const char* key,
                                    const char* description) {
    const auto node = object[key];
    if (node.empty()) {
        invalid(std::string("missing ") + description);
    }
    return node;
}

[[nodiscard]] bool has_key(const cv::FileNode& object, const char* key) {
    for (const auto& field : object) {
        if (field.name() == key) {
            return true;
        }
    }
    return false;
}

void validate_keys(const cv::FileNode& object,
                   std::initializer_list<std::string_view> allowed,
                   const char* description) {
    if (!object.isMap()) {
        invalid(std::string(description) + " must be an object");
    }
    std::set<std::string> seen;
    for (const auto& field : object) {
        const std::string key { field.name() };
        if (!seen.insert(key).second) {
            invalid(std::string(description) + " contains a duplicate key: " + key);
        }
        if (std::find(allowed.begin(), allowed.end(), key) == allowed.end()) {
            invalid(std::string(description) + " contains an unknown key: " + key);
        }
    }
}

[[nodiscard]] std::string read_string(const cv::FileNode& node,
                                      const char* description) {
    if (!node.isString()) {
        invalid(std::string(description) + " must be a string");
    }
    std::string value;
    node >> value;
    if (value.empty()) {
        invalid(std::string(description) + " must not be empty");
    }
    return value;
}

[[nodiscard]] double read_number(const cv::FileNode& node,
                                 const char* description) {
    if (!is_number(node)) {
        invalid(std::string(description) + " must be a number");
    }
    double value = 0.0;
    node >> value;
    if (!std::isfinite(value)) {
        invalid(std::string(description) + " must be finite");
    }
    return value;
}

[[nodiscard]] int read_integer(const cv::FileNode& node,
                               const char* description) {
    if (!node.isInt()) {
        invalid(std::string(description) + " must be an integer");
    }
    int value = 0;
    node >> value;
    return value;
}

[[nodiscard]] bool read_boolean(const cv::FileNode& node,
                                const char* description) {
    if (!node.isInt()) {
        invalid(std::string(description) + " must be a boolean");
    }
    int value = 0;
    node >> value;
    if (value != 0 && value != 1) {
        invalid(std::string(description) + " must be a boolean");
    }
    return value != 0;
}

[[nodiscard]] std::filesystem::path resolve_relative_path(
    const std::filesystem::path& config_path,
    const cv::FileNode& node,
    const char* description) {
    const std::filesystem::path relative { read_string(node, description) };
    if (relative.is_absolute() || relative.has_root_name() || relative.has_root_directory()) {
        invalid(std::string(description) + " must be relative to the configuration file");
    }
    const auto parent = config_path.parent_path().empty()
                            ? std::filesystem::current_path()
                            : std::filesystem::absolute(config_path.parent_path());
    return (parent / relative).lexically_normal();
}

[[nodiscard]] std::filesystem::path read_metadata_path(
    const std::filesystem::path& config_path,
    const cv::FileNode& node,
    const char* description) {
    const auto path = resolve_relative_path(config_path, node, description);
    std::error_code error;
    const auto status = std::filesystem::status(path, error);
    constexpr auto readable = std::filesystem::perms::owner_read
                              | std::filesystem::perms::group_read
                              | std::filesystem::perms::others_read;
    if (error || !std::filesystem::is_regular_file(status)
        || (status.permissions() & readable) == std::filesystem::perms::none) {
        invalid(std::string(description) + " must be a readable regular file: " + path.string());
    }
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        invalid(std::string(description) + " must be a readable regular file: " + path.string());
    }
    return path;
}

[[nodiscard]] bool valid_frame_prefix(const std::string& value) noexcept;

void read_nested_options(const std::filesystem::path& config_path,
                         const cv::FileNode& root,
                         EngineOptions& engine) {
    if (has_key(root, "modalities")) {
        const auto node = root["modalities"];
        validate_keys(node, { "region", "depth", "texture" }, "modalities");
        if (has_key(node, "region")) {
            engine.use_region = read_boolean(node["region"], "modalities.region");
        }
        if (has_key(node, "depth")) {
            engine.use_depth = read_boolean(node["depth"], "modalities.depth");
        }
        if (has_key(node, "texture")) {
            engine.use_texture = read_boolean(node["texture"], "modalities.texture");
        }
        if (!engine.use_region && !engine.use_depth && !engine.use_texture) {
            invalid("at least one modality must be enabled");
        }
    }

    if (has_key(root, "occlusion")) {
        const auto node = root["occlusion"];
        validate_keys(node, { "measured", "modeled", "region_checking", "image_size" },
                      "occlusion");
        if (has_key(node, "measured")) {
            engine.measured_occlusions = read_boolean(node["measured"], "occlusion.measured");
        }
        if (has_key(node, "modeled")) {
            engine.modeled_occlusions = read_boolean(node["modeled"], "occlusion.modeled");
        }
        if (has_key(node, "region_checking")) {
            engine.region_checking = read_boolean(node["region_checking"],
                                                  "occlusion.region_checking");
        }
        if (has_key(node, "image_size")) {
            engine.occlusion_image_size = read_integer(node["image_size"], "occlusion.image_size");
            if (engine.occlusion_image_size < 64 || engine.occlusion_image_size > 2048) {
                invalid("occlusion.image_size is outside the supported range");
            }
        }
    }

    if (has_key(root, "metadata")) {
        const auto node = root["metadata"];
        validate_keys(node, { "region", "depth", "texture", "optimizer", "refiner" },
                      "metadata");
        if (has_key(node, "region")) {
            engine.region_metadata = read_metadata_path(config_path, node["region"],
                                                        "metadata.region");
        }
        if (has_key(node, "depth")) {
            engine.depth_metadata = read_metadata_path(config_path, node["depth"],
                                                       "metadata.depth");
        }
        if (has_key(node, "texture")) {
            engine.texture_metadata = read_metadata_path(config_path, node["texture"],
                                                         "metadata.texture");
        }
        if (has_key(node, "optimizer")) {
            engine.optimizer_metadata = read_metadata_path(config_path, node["optimizer"],
                                                           "metadata.optimizer");
        }
        if (has_key(node, "refiner")) {
            engine.refiner_metadata = read_metadata_path(config_path, node["refiner"],
                                                         "metadata.refiner");
        }
    }

    if (has_key(root, "initialization_flow")) {
        const auto node = root["initialization_flow"];
        validate_keys(node, { "static_detector", "refine" }, "initialization_flow");
        if (has_key(node, "static_detector")) {
            engine.use_static_detector = read_boolean(node["static_detector"],
                                                      "initialization_flow.static_detector");
        }
        if (has_key(node, "refine")) {
            engine.refine_initialization = read_boolean(node["refine"],
                                                        "initialization_flow.refine");
        }
    }

    if (has_key(root, "mount_refinement")) {
        const auto node = root["mount_refinement"];
        validate_keys(node, { "enabled", "translation_standard_deviation_m",
                              "rotation_standard_deviation_rad" }, "mount_refinement");
        auto& refinement = engine.mount_refinement;
        if (has_key(node, "enabled")) {
            refinement.enabled = read_boolean(node["enabled"], "mount_refinement.enabled");
        }
        const auto sigma = [&node](const char* key, double& target, double maximum) {
            if (has_key(node, key)) {
                target = read_number(node[key], key);
                if (target < 0.0001 || target > maximum) {
                    invalid(std::string("mount_refinement.") + key
                            + " is outside the supported range");
                }
            }
        };
        sigma("translation_standard_deviation_m", refinement.translation_standard_deviation_m,
              0.2);
        sigma("rotation_standard_deviation_rad", refinement.rotation_standard_deviation_rad,
              0.785398);
    }

    if (has_key(root, "visualization")) {
        const auto node = root["visualization"];
        validate_keys(node, { "correspondences", "optimization", "results" }, "visualization");
        if (has_key(node, "correspondences")) {
            engine.visualize_correspondences = read_boolean(node["correspondences"],
                                                            "visualization.correspondences");
        }
        if (has_key(node, "optimization")) {
            engine.visualize_optimization = read_boolean(node["optimization"],
                                                         "visualization.optimization");
        }
        if (has_key(node, "results")) {
            engine.visualize_results = read_boolean(node["results"], "visualization.results");
        }
    }
}

[[nodiscard]] JointPriorConfig read_joint_prior_config(const cv::FileNode& root) {
    JointPriorConfig config;
    if (!has_key(root, "joint_priors")) {
        return config;
    }

    const auto node = root["joint_priors"];
    validate_keys(node,
                  { "enabled", "required", "initialize_from_measurement", "service_name",
                    "controller_id", "time_basis", "maximum_age_ms", "capacity", "mappings" },
                  "joint_priors");
    config.enabled = true;
    if (has_key(node, "enabled")) {
        config.enabled = read_boolean(node["enabled"], "joint_priors.enabled");
    }
    if (has_key(node, "required")) {
        config.required = read_boolean(node["required"], "joint_priors.required");
    }
    if (has_key(node, "initialize_from_measurement")) {
        config.initialize_from_measurement = read_boolean(
            node["initialize_from_measurement"], "joint_priors.initialize_from_measurement");
    }
    if (has_key(node, "service_name")) {
        config.service_name = read_string(node["service_name"], "joint_priors.service_name");
        if (config.service_name.size() > 240U) {
            invalid("joint_priors.service_name must be at most 240 bytes");
        }
    }
    if (has_key(node, "controller_id")) {
        config.controller_id = read_string(node["controller_id"], "joint_priors.controller_id");
        if (config.controller_id.size() > 128U) {
            invalid("joint_priors.controller_id must be at most 128 bytes");
        }
    }
    if (has_key(node, "time_basis")) {
        config.time_basis = read_string(node["time_basis"], "joint_priors.time_basis");
        if (config.time_basis != "capture" && config.time_basis != "received") {
            invalid("joint_priors.time_basis must be capture or received");
        }
    }
    if (has_key(node, "maximum_age_ms")) {
        const int maximum_age_ms = read_integer(node["maximum_age_ms"],
                                                "joint_priors.maximum_age_ms");
        if (maximum_age_ms < 1 || maximum_age_ms > 1000) {
            invalid("joint_priors.maximum_age_ms is outside the supported range");
        }
        constexpr std::uint64_t nanoseconds_per_millisecond = 1'000'000U;
        config.maximum_age_ns = static_cast<std::uint64_t>(maximum_age_ms)
                                * nanoseconds_per_millisecond;
    }
    if (has_key(node, "capacity")) {
        const int capacity = read_integer(node["capacity"], "joint_priors.capacity");
        if (capacity < 1 || capacity > 512) {
            invalid("joint_priors.capacity is outside the supported range");
        }
        config.capacity = static_cast<std::size_t>(capacity);
    }

    if (has_key(node, "mappings")) {
        const auto mappings = node["mappings"];
        if (!mappings.isSeq() || mappings.size() < 1U || mappings.size() > 64U) {
            invalid("joint_priors.mappings must contain between one and 64 entries");
        }
        config.mappings.reserve(mappings.size());
        for (const auto& entry : mappings) {
            validate_keys(entry,
                          { "arm", "joint", "source_arm", "source_joint",
                            "scale_rad_per_unit", "offset_rad", "standard_deviation_rad" },
                          "joint-prior mapping");
            JointPriorMapping mapping;
            mapping.arm = read_string(required(entry, "arm", "joint-prior mapping arm"),
                                      "joint-prior mapping arm");
            mapping.joint = read_string(required(entry, "joint", "joint-prior mapping joint"),
                                        "joint-prior mapping joint");
            mapping.source_arm = read_string(
                required(entry, "source_arm", "joint-prior mapping source_arm"),
                "joint-prior mapping source_arm");
            mapping.source_joint = read_string(
                required(entry, "source_joint", "joint-prior mapping source_joint"),
                "joint-prior mapping source_joint");
            mapping.scale_rad_per_unit = read_number(
                required(entry, "scale_rad_per_unit", "joint-prior mapping scale_rad_per_unit"),
                "joint-prior mapping scale_rad_per_unit");
            mapping.offset_rad = read_number(
                required(entry, "offset_rad", "joint-prior mapping offset_rad"),
                "joint-prior mapping offset_rad");
            mapping.standard_deviation_rad = read_number(
                required(entry, "standard_deviation_rad",
                         "joint-prior mapping standard_deviation_rad"),
                "joint-prior mapping standard_deviation_rad");
            if (mapping.scale_rad_per_unit == 0.0) {
                invalid("joint-prior mapping scale_rad_per_unit must be nonzero");
            }
            if (mapping.standard_deviation_rad <= 0.0) {
                invalid("joint-prior mapping standard_deviation_rad must be positive");
            }
            config.mappings.push_back(std::move(mapping));
        }
    }

    if (config.enabled) {
        if (config.service_name.empty()) {
            invalid("joint_priors.service_name is required when enabled");
        }
        if (config.controller_id.empty()) {
            invalid("joint_priors.controller_id is required when enabled");
        }
        if (config.mappings.empty()) {
            invalid("joint_priors.mappings is required when enabled");
        }
    }
    return config;
}

[[nodiscard]] std::vector<ConstraintOptions> read_constraints(
    const std::filesystem::path& config_path,
    const cv::FileNode& root,
    const RobotModel& model,
    const std::vector<ArmSeed>& arms) {
    if (!has_key(root, "constraints")) {
        return {};
    }
    const auto node = root["constraints"];
    if (!node.isSeq()) {
        invalid("constraints must be an array");
    }
    if (node.size() > kMaximumConstraints) {
        invalid("constraints must contain at most 64 entries");
    }

    std::set<std::string> known_arms;
    for (const auto& arm : arms) {
        known_arms.insert(arm.name);
    }
    std::set<std::string> known_links;
    for (const auto& link : model.links) {
        known_links.insert(link.name);
    }
    std::set<std::string> names;
    std::vector<ConstraintOptions> constraints;
    constraints.reserve(node.size());
    for (const auto& entry : node) {
        validate_keys(entry, { "name", "arm", "link1", "link2", "type", "metadata" },
                      "constraint");
        ConstraintOptions constraint;
        constraint.name = read_string(required(entry, "name", "constraint name"),
                                      "constraint name");
        if (constraint.name.size() > kMaximumNameBytes) {
            invalid("constraint name must be at most 256 bytes");
        }
        if (!names.insert(constraint.name).second) {
            invalid("constraint names must be unique");
        }
        constraint.arm = read_string(required(entry, "arm", "constraint arm"),
                                     "constraint arm");
        if (!valid_frame_prefix(constraint.arm)
            || known_arms.find(constraint.arm) == known_arms.end()) {
            invalid("constraint arm must name a configured arm");
        }
        constraint.link1 = read_string(required(entry, "link1", "constraint link1"),
                                       "constraint link1");
        constraint.link2 = read_string(required(entry, "link2", "constraint link2"),
                                       "constraint link2");
        if (known_links.find(constraint.link1) == known_links.end()
            || known_links.find(constraint.link2) == known_links.end()) {
            invalid("constraint links must name links in the robot model");
        }
        if (constraint.link1 == constraint.link2) {
            invalid("constraint links must be different");
        }
        const auto type = read_string(required(entry, "type", "constraint type"),
                                      "constraint type");
        if (type != "hard" && type != "soft") {
            invalid("constraint type must be hard or soft");
        }
        constraint.soft = type == "soft";
        constraint.metadata = read_metadata_path(config_path,
                                                 required(entry, "metadata", "constraint metadata"),
                                                 "constraint metadata");
        constraints.push_back(std::move(constraint));
    }
    return constraints;
}

[[nodiscard]] cv::FileStorage open_json(const std::filesystem::path& path,
                                        const char* description) {
    cv::FileStorage storage;
    try {
        if (!storage.open(path.string(), cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON)) {
            invalid(std::string("could not open ") + description + ": " + path.string());
        }
    } catch (const cv::Exception& error) {
        invalid(std::string("could not parse ") + description + ": " + error.what());
    }
    return storage;
}

[[nodiscard]] bool valid_frame_prefix(const std::string& value) noexcept {
    if (value.find('/') != std::string::npos) {
        return false;
    }
    try {
        (void)rbt::tf::FrameId(value);
        return true;
    } catch (const rbt::tf::TfError&) {
        return false;
    }
}

void validate_visual_link_weight(const std::string& link, double weight) {
    if (link.empty() || link.size() > kMaximumNameBytes || !valid_frame_prefix(link)
        || !std::isfinite(weight) || weight < 0.01 || weight > 100.0) {
        invalid("visual_link_weights requires valid link names and finite weights in [0.01, 100]");
    }
}

[[nodiscard]] std::map<std::string, double> read_visual_link_weights(
    const cv::FileNode& root, const RobotModel& model) {
    std::map<std::string, double> weights;
    if (!has_key(root, "visual_link_weights")) {
        return weights;
    }
    const auto node = root["visual_link_weights"];
    if (!node.isMap()) {
        invalid("visual_link_weights must be an object");
    }
    for (const auto& field : node) {
        const std::string name = field.name();
        const double weight = read_number(field, "visual_link_weights weight");
        validate_visual_link_weight(name, weight);
        const auto link = std::find_if(model.links.begin(), model.links.end(),
            [&name](const ModelLink& candidate) { return candidate.name == name; });
        if (link == model.links.end() || link->mesh.empty()) {
            invalid("visual_link_weights must name a model link with a mesh");
        }
        if (!weights.emplace(name, weight).second) {
            invalid("visual_link_weights contains a duplicate link name");
        }
    }
    return weights;
}

void append_visual_link_weights_json(std::ostringstream& output,
                                     const std::map<std::string, double>& weights) {
    if (weights.empty()) {
        return;
    }
    output << ",\"visual_link_weights\":{";
    bool first = true;
    for (const auto& [link, weight] : weights) {
        validate_visual_link_weight(link, weight);
        if (!first) {
            output << ',';
        }
        first = false;
        output << rbt_orbbec::json_quote(link) << ':' << weight;
    }
    output << '}';
}

[[nodiscard]] bool is_rigid_pose(const Eigen::Affine3f& pose) noexcept {
    const Eigen::Matrix4f matrix = pose.matrix();
    if (!matrix.array().isFinite().all()) {
        return false;
    }
    if ((matrix.row(3) - Eigen::RowVector4f { 0.0F, 0.0F, 0.0F, 1.0F })
            .cwiseAbs()
            .maxCoeff()
        > kRigidTolerance) {
        return false;
    }
    const Eigen::Matrix3f rotation = matrix.topLeftCorner<3, 3>();
    if ((rotation.transpose() * rotation - Eigen::Matrix3f::Identity())
            .cwiseAbs()
            .maxCoeff()
        > kRigidTolerance) {
        return false;
    }
    return std::abs(rotation.determinant() - 1.0F) <= kRigidTolerance;
}

[[nodiscard]] Eigen::Affine3f read_pose(const cv::FileNode& node,
                                        const char* description) {
    if (!node.isSeq() || node.size() != 16U) {
        invalid(std::string(description) + " must contain exactly 16 row-major numbers");
    }
    Eigen::Affine3f pose = Eigen::Affine3f::Identity();
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            const auto value = read_number(node[row * 4 + column], description);
            if (std::abs(value) > std::numeric_limits<float>::max()) {
                invalid(std::string(description) + " exceeds the supported range");
            }
            pose.matrix()(row, column) = static_cast<float>(value);
        }
    }
    if (!is_rigid_pose(pose)) {
        invalid(std::string(description) + " must be a finite proper rigid transform");
    }
    return pose;
}

[[nodiscard]] CameraMotionConfig read_camera_motion_config(
    const cv::FileNode& root,
    const std::string& initialization_frame,
    const std::vector<ArmSeed>& arms) {
    CameraMotionConfig config;
    if (!has_key(root, "camera_motion")) {
        return config;
    }

    const auto node = root["camera_motion"];
    validate_keys(node,
                  { "enabled", "world_frame", "pitch_frame", "pitch_provider", "depth_frame",
                    "time_basis", "allow_unknown_uncertainty", "maximum_interpolation_gap_ms",
                    "maximum_wait_ms", "maximum_sample_age_ms", "mount_in_pitch", "arm_bases" },
                  "camera_motion");
    config.enabled = true;
    if (has_key(node, "enabled")) {
        config.enabled = read_boolean(node["enabled"], "camera_motion.enabled");
    }
    if (has_key(node, "world_frame")) {
        config.world_frame = read_string(node["world_frame"], "camera_motion.world_frame");
    }
    if (has_key(node, "pitch_frame")) {
        config.pitch_frame = read_string(node["pitch_frame"], "camera_motion.pitch_frame");
    }
    if (has_key(node, "pitch_provider")) {
        config.pitch_provider = read_string(node["pitch_provider"], "camera_motion.pitch_provider");
    }
    if (has_key(node, "depth_frame")) {
        config.depth_frame = read_string(node["depth_frame"], "camera_motion.depth_frame");
    }
    if (has_key(node, "time_basis")) {
        config.time_basis = read_string(node["time_basis"], "camera_motion.time_basis");
    }
    if (has_key(node, "allow_unknown_uncertainty")) {
        config.allow_unknown_uncertainty = read_boolean(
            node["allow_unknown_uncertainty"], "camera_motion.allow_unknown_uncertainty");
    }
    if (has_key(node, "maximum_interpolation_gap_ms")) {
        config.maximum_interpolation_gap_ms = read_integer(
            node["maximum_interpolation_gap_ms"], "camera_motion.maximum_interpolation_gap_ms");
    }
    if (has_key(node, "maximum_wait_ms")) {
        config.maximum_wait_ms = read_integer(node["maximum_wait_ms"],
                                              "camera_motion.maximum_wait_ms");
    }
    if (has_key(node, "maximum_sample_age_ms")) {
        config.maximum_sample_age_ms = read_integer(node["maximum_sample_age_ms"],
                                                    "camera_motion.maximum_sample_age_ms");
    }
    if (has_key(node, "mount_in_pitch")) {
        config.mount_in_pitch = read_pose(node["mount_in_pitch"], "camera_motion.mount_in_pitch");
    }
    if (has_key(node, "arm_bases")) {
        const auto entries = node["arm_bases"];
        if (!entries.isSeq() || entries.size() > 64U) {
            invalid("camera_motion.arm_bases must be an array with at most 64 entries");
        }
        std::set<std::string> names;
        config.arm_bases.reserve(entries.size());
        for (const auto& entry : entries) {
            validate_keys(entry, { "arm", "base_in_world" }, "camera-motion arm base");
            CameraMotionArmBase arm_base;
            arm_base.arm = read_string(required(entry, "arm", "camera-motion arm base arm"),
                                       "camera-motion arm base arm");
            if (!valid_frame_prefix(arm_base.arm) || !names.insert(arm_base.arm).second) {
                invalid("camera_motion.arm_bases arm names must be valid and unique");
            }
            arm_base.base_in_world = read_pose(
                required(entry, "base_in_world", "camera-motion base_in_world"),
                "camera_motion arm base_in_world");
            config.arm_bases.push_back(std::move(arm_base));
        }
    }

    if (config.maximum_interpolation_gap_ms < 1
        || config.maximum_interpolation_gap_ms > 1000) {
        invalid("camera_motion.maximum_interpolation_gap_ms is outside the supported range");
    }
    if (config.maximum_wait_ms < 0 || config.maximum_wait_ms > 1000) {
        invalid("camera_motion.maximum_wait_ms is outside the supported range");
    }
    if (config.maximum_sample_age_ms < 1 || config.maximum_sample_age_ms > 10'000) {
        invalid("camera_motion.maximum_sample_age_ms is outside the supported range");
    }
    if (config.time_basis != "capture" && config.time_basis != "received") {
        invalid("camera_motion.time_basis must be capture or received");
    }
    if (!config.enabled) {
        return config;
    }

    if (config.world_frame.empty() || config.pitch_frame.empty()
        || config.pitch_provider.empty() || config.depth_frame.empty()
        || !has_key(node, "mount_in_pitch") || !has_key(node, "arm_bases")) {
        invalid("camera_motion requires world, pitch, provider, depth, mount, and arm bases when enabled");
    }
    try {
        (void)rbt::tf::FrameId(config.world_frame);
        (void)rbt::tf::FrameId(config.pitch_frame);
        (void)rbt::tf::FrameId(config.depth_frame);
        (void)rbt::tf::FrameId(config.pitch_provider);
    } catch (const rbt::tf::TfError&) {
        invalid("camera_motion frames and provider must be valid RBT identifiers");
    }
    if (config.world_frame == config.pitch_frame || config.world_frame == config.depth_frame
        || config.pitch_frame == config.depth_frame) {
        invalid("camera_motion world_frame, pitch_frame, and depth_frame must be distinct");
    }
    if (config.depth_frame == initialization_frame) {
        invalid("camera_motion.depth_frame must differ from initialization coordinate_frame");
    }
    if (!is_rigid_pose(config.mount_in_pitch)) {
        invalid("camera_motion.mount_in_pitch must be a finite proper rigid transform");
    }

    std::set<std::string> expected_arms;
    for (const auto& arm : arms) {
        if (!arm.fixed_base) {
            invalid("camera_motion requires all configured arms to have fixed_base=true");
        }
        expected_arms.insert(arm.name);
    }
    std::set<std::string> configured_arms;
    for (const auto& arm_base : config.arm_bases) {
        if (!is_rigid_pose(arm_base.base_in_world)) {
            invalid("camera_motion arm base must be a finite proper rigid transform");
        }
        configured_arms.insert(arm_base.arm);
    }
    if (configured_arms != expected_arms) {
        invalid("camera_motion.arm_bases must match the configured arm set exactly");
    }
    return config;
}

[[nodiscard]] std::optional<SecondaryCameraConfig> read_secondary_camera_config(
    const cv::FileNode& root,
    const std::string& primary_frame,
    const std::string& primary_serial) {
    if (!has_key(root, "secondary_camera")) {
        return std::nullopt;
    }

    const auto node = root["secondary_camera"];
    validate_keys(node,
                  { "provider", "serial", "coordinate_frame", "camera_in_primary",
                    "maximum_skew_ms", "time_basis" },
                  "secondary_camera");

    SecondaryCameraConfig config;
    config.provider = read_string(required(node, "provider", "secondary_camera.provider"),
                                  "secondary_camera.provider");
    config.serial = read_string(required(node, "serial", "secondary_camera.serial"),
                                "secondary_camera.serial");
    config.coordinate_frame = read_string(
        required(node, "coordinate_frame", "secondary_camera.coordinate_frame"),
        "secondary_camera.coordinate_frame");
    config.camera_in_primary = read_pose(
        required(node, "camera_in_primary", "secondary_camera.camera_in_primary"),
        "secondary_camera.camera_in_primary");

    if (config.provider.size() > 63U || !valid_frame_prefix(config.provider)) {
        invalid("secondary_camera.provider must be a valid single-segment RBT identifier");
    }
    if (config.serial.size() > kMaximumNameBytes) {
        invalid("secondary_camera.serial must be at most 256 bytes");
    }
    try {
        (void)rbt::tf::FrameId(config.coordinate_frame);
    } catch (const rbt::tf::TfError&) {
        invalid("secondary_camera.coordinate_frame must be a valid RBT frame id");
    }
    if (config.serial == primary_serial) {
        invalid("secondary_camera.serial must differ from the primary camera serial");
    }
    if (config.coordinate_frame == primary_frame) {
        invalid("secondary_camera.coordinate_frame must differ from the primary camera frame");
    }

    if (has_key(node, "maximum_skew_ms")) {
        const int maximum_skew_ms = read_integer(node["maximum_skew_ms"],
                                                 "secondary_camera.maximum_skew_ms");
        if (maximum_skew_ms < 1 || maximum_skew_ms > 100) {
            invalid("secondary_camera.maximum_skew_ms is outside the supported range");
        }
        config.maximum_skew_ms = static_cast<std::uint64_t>(maximum_skew_ms);
    }
    if (has_key(node, "time_basis")) {
        config.time_basis = read_string(node["time_basis"], "secondary_camera.time_basis");
    }
    if (config.time_basis != "capture" && config.time_basis != "received") {
        invalid("secondary_camera.time_basis must be capture or received");
    }
    return config;
}

[[nodiscard]] std::vector<ArmSeed> read_arms(const cv::FileNode& node,
                                            const RobotModel& model) {
    if (!node.isSeq() || node.size() == 0U || node.size() > 2U) {
        invalid("initialization arms must contain one or two entries");
    }

    std::set<std::string> arm_names;
    std::set<std::string> expected_joints;
    for (const auto& joint : model.joints) {
        if (joint.type == "revolute") {
            expected_joints.insert(joint.name);
        }
    }

    std::vector<ArmSeed> arms;
    arms.reserve(node.size());
    for (const auto& arm_node : node) {
        validate_keys(arm_node, { "name", "base_in_camera", "joint_angles", "fixed_base" },
                      "initialization arm");
        ArmSeed arm;
        arm.name = read_string(required(arm_node, "name", "arm name"), "arm name");
        if (!valid_frame_prefix(arm.name)) {
            invalid("arm name must be a valid RBT frame prefix");
        }
        if (!arm_names.insert(arm.name).second) {
            invalid("initialization arm names must be unique");
        }
        arm.base_in_camera = read_pose(required(arm_node, "base_in_camera", "arm base_in_camera"),
                                       "arm base_in_camera");
        arm.fixed_base = read_boolean(required(arm_node, "fixed_base", "arm fixed_base"),
                                      "arm fixed_base");

        const auto angles_node = required(arm_node, "joint_angles", "arm joint_angles");
        if (!angles_node.isMap()) {
            invalid("arm joint_angles must be an object");
        }
        std::set<std::string> supplied_joints;
        for (const auto& angle_node : angles_node) {
            const std::string joint_name { angle_node.name() };
            if (!supplied_joints.insert(joint_name).second) {
                invalid("arm joint_angles contains a duplicate joint: " + joint_name);
            }
            if (expected_joints.find(joint_name) == expected_joints.end()) {
                invalid("arm joint_angles contains an unknown or fixed joint: " + joint_name);
            }
            const auto angle = read_number(angle_node, "joint angle");
            const auto joint = std::find_if(model.joints.begin(), model.joints.end(),
                                            [&joint_name](const ModelJoint& candidate) {
                                                return candidate.name == joint_name;
                                            });
            if (angle < joint->lower || angle > joint->upper) {
                invalid("arm joint angle is outside the URDF joint limits: " + joint_name);
            }
            arm.joint_angles.emplace(joint_name, angle);
        }
        if (supplied_joints != expected_joints) {
            invalid("arm joint_angles must provide exactly every revolute joint angle");
        }
        arms.push_back(std::move(arm));
    }
    return arms;
}

struct Initialization {
    std::string coordinate_frame;
    std::string serial;
    std::vector<ArmSeed> arms;
};

[[nodiscard]] Initialization read_initialization(const std::filesystem::path& path,
                                                 const RobotModel& model) {
    auto storage = open_json(path, "initialization JSON");
    const auto root = storage.root();
    validate_keys(root, { "schema_version", "coordinate_frame", "serial", "arms", "provenance" },
                  "initialization JSON");
    if (read_integer(required(root, "schema_version", "initialization schema_version"),
                     "initialization schema_version") != 1) {
        invalid("unsupported initialization schema_version");
    }

    auto coordinate_frame = read_string(required(root, "coordinate_frame", "coordinate_frame"),
                                         "coordinate_frame");
    try {
        (void)rbt::tf::FrameId(coordinate_frame);
    } catch (const rbt::tf::TfError&) {
        invalid("coordinate_frame must be a valid RBT frame id");
    }
    auto serial = read_string(required(root, "serial", "serial"), "serial");
    auto arms = read_arms(required(root, "arms", "initialization arms"), model);
    return Initialization { std::move(coordinate_frame), std::move(serial), std::move(arms) };
}

[[nodiscard]] std::uint64_t timestamp_distance(std::int64_t lhs,
                                               std::int64_t rhs) noexcept {
    if (lhs >= rhs) {
        return static_cast<std::uint64_t>(lhs) - static_cast<std::uint64_t>(rhs);
    }
    return static_cast<std::uint64_t>(rhs) - static_cast<std::uint64_t>(lhs);
}

void append_pose(std::ostringstream& output, const Eigen::Affine3f& pose);
[[nodiscard]] const char* clock_kind_name(rbt::time::ClockKind kind) noexcept;

void append_timestamp_json(std::ostringstream& output,
                           const rbt::time::Timestamp& timestamp) {
    if (!timestamp.clock.is_well_formed() || timestamp.ns <= 0) {
        invalid("secondary-camera timing has an invalid clock identity");
    }
    output << "{\"timestamp_ns\":" << timestamp.ns
           << ",\"clock_kind\":"
           << rbt_orbbec::json_quote(clock_kind_name(timestamp.clock.kind()))
           << ",\"clock_source\":"
           << rbt_orbbec::json_quote(timestamp.clock.source_id().to_hex())
           << ",\"clock_generation\":" << timestamp.clock.generation() << '}';
}

void append_timing_json(std::ostringstream& output, const RectifiedPair& pair) {
    if (!pair.timing.capture.has_value()) {
        invalid("secondary-camera tracking output requires capture timing");
    }
    output << "{\"capture\":";
    append_timestamp_json(output, *pair.timing.capture);
    output << ",\"received\":";
    append_timestamp_json(output, pair.timing.received);
    output << '}';
}

void append_secondary_camera_json(std::ostringstream& output,
                                  const RectifiedPair& primary,
                                  const RectifiedPair& secondary,
                                  const SecondaryCameraConfig& config) {
    if (config.time_basis != "capture" && config.time_basis != "received") {
        invalid("secondary-camera configuration has an invalid time basis");
    }
    if (config.maximum_skew_ms < 1U || config.maximum_skew_ms > 100U) {
        invalid("secondary-camera maximum skew is outside the supported range");
    }
    if (config.provider.empty() || config.serial.empty()
        || config.serial.size() > kMaximumNameBytes
        || config.coordinate_frame == primary.frame_id) {
        invalid("secondary-camera configuration contains a missing or duplicate identity");
    }
    if (secondary.frame_id != config.coordinate_frame) {
        invalid("secondary-camera pair frame does not match configured coordinate_frame");
    }
    if (!primary.timing.capture.has_value() || !secondary.timing.capture.has_value()) {
        invalid("dual-camera tracking output requires capture timestamps for both cameras");
    }
    const auto& primary_capture = *primary.timing.capture;
    const auto& secondary_capture = *secondary.timing.capture;
    if (!supported_capture_clock(primary_capture.clock)
        || !supported_capture_clock(secondary_capture.clock)
        || !primary.color_capture.clock.is_well_formed()
        || !secondary.color_capture.clock.is_well_formed()
        || primary_capture.clock != primary.color_capture.clock
        || secondary_capture.clock != secondary.color_capture.clock
        || primary.skew_ns != timestamp_distance(primary_capture.ns, primary.color_capture.ns)
        || secondary.skew_ns != timestamp_distance(secondary_capture.ns, secondary.color_capture.ns)) {
        invalid("dual-camera output pairs have inconsistent capture clocks or timestamps");
    }
    const auto& primary_stamp = config.time_basis == "capture"
                                    ? primary_capture
                                    : primary.timing.received;
    const auto& secondary_stamp = config.time_basis == "capture"
                                      ? secondary_capture
                                      : secondary.timing.received;
    if (!primary_stamp.clock.is_well_formed() || !secondary_stamp.clock.is_well_formed()
        || primary_stamp.ns <= 0 || secondary_stamp.ns <= 0
        || primary_stamp.clock != secondary_stamp.clock
        || (config.time_basis == "received"
            && (primary_stamp.clock.kind() != rbt::time::ClockKind::HostMonotonic
                || secondary_stamp.clock.kind() != rbt::time::ClockKind::HostMonotonic))) {
        invalid("dual-camera selected timestamps do not share a complete clock identity");
    }
    const auto skew_ns = timestamp_distance(primary_stamp.ns, secondary_stamp.ns);
    const auto maximum_skew_ns = config.maximum_skew_ms * 1'000'000U;
    if (skew_ns > maximum_skew_ns) {
        invalid("dual-camera selected timestamps exceed the configured maximum skew");
    }
    try {
        if (!valid_frame_prefix(config.provider)) {
            invalid("secondary-camera provider must be a valid RBT identifier");
        }
        (void)rbt::tf::FrameId(config.coordinate_frame);
    } catch (const rbt::tf::TfError&) {
        invalid("secondary-camera configuration contains an invalid identifier");
    }

    output << ",\"secondary_camera\":{\"provider\":"
           << rbt_orbbec::json_quote(config.provider)
           << ",\"serial\":" << rbt_orbbec::json_quote(config.serial)
           << ",\"frame_id\":" << rbt_orbbec::json_quote(config.coordinate_frame)
           << ",\"time_basis\":" << rbt_orbbec::json_quote(config.time_basis)
           << ",\"maximum_skew_ns\":" << maximum_skew_ns
           << ",\"cross_view_skew_ns\":" << skew_ns
           << ",\"camera_in_primary\":";
    append_pose(output, config.camera_in_primary);
    output << ",\"primary_timing\":";
    append_timing_json(output, primary);
    output << ",\"secondary_timing\":";
    append_timing_json(output, secondary);
    output << '}';
}

void append_pose(std::ostringstream& output, const Eigen::Affine3f& pose) {
    if (!is_rigid_pose(pose)) {
        invalid("tracking estimate contains a non-finite or non-rigid pose");
    }
    output << '[';
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            if (row != 0 || column != 0) {
                output << ',';
            }
            output << pose.matrix()(row, column);
        }
    }
    output << ']';
}

[[nodiscard]] const char* clock_kind_name(rbt::time::ClockKind kind) noexcept {
    switch (kind) {
    case rbt::time::ClockKind::HostMonotonic:
        return "HostMonotonic";
    case rbt::time::ClockKind::SystemRealtime:
        return "SystemRealtime";
    case rbt::time::ClockKind::Device:
        return "Device";
    }
    return "Unknown";
}

void append_camera_motion_json(std::ostringstream& output,
                               const RectifiedPair& pair,
                               const CameraMotionSelection& motion) {
    if (motion.time_basis != "capture" && motion.time_basis != "received") {
        invalid("camera-motion selection has an invalid time basis");
    }
    try {
        (void)rbt::tf::FrameId(motion.world_frame);
    } catch (const rbt::tf::TfError&) {
        invalid("camera-motion selection has an invalid world frame");
    }
    const auto& expected_stamp = motion.time_basis == "capture"
                                     ? *pair.timing.capture
                                     : pair.timing.received;
    if (motion.lookup.requested_timestamp.clock != expected_stamp.clock
        || motion.lookup.requested_timestamp.ns != expected_stamp.ns) {
        invalid("camera-motion selection timestamp does not match the output pair");
    }
    if (motion.selection_policy != "nearest"
        || !motion.pitch_sample_timestamp.clock.is_well_formed()
        || motion.pitch_sample_timestamp.clock != expected_stamp.clock) {
        invalid("camera-motion selection has invalid pitch-sample provenance");
    }
    const auto actual_abs_offset = timestamp_distance(motion.pitch_sample_timestamp.ns,
                                                      expected_stamp.ns);
    if (actual_abs_offset > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        invalid("camera-motion pitch-sample offset is outside the supported range");
    }
    const auto actual_offset = motion.pitch_sample_timestamp.ns >= expected_stamp.ns
                                   ? static_cast<std::int64_t>(actual_abs_offset)
                                   : -static_cast<std::int64_t>(actual_abs_offset);
    if (motion.pitch_sample_offset_abs_ns != actual_abs_offset
        || motion.pitch_sample_offset_ns != actual_offset) {
        invalid("camera-motion pitch-sample offset does not match its timestamps");
    }

    output << ",\"camera_motion\":{\"enabled\":true,\"status\":\"used\",\"used\":true"
           << ",\"world_frame\":" << rbt_orbbec::json_quote(motion.world_frame)
           << ",\"time_basis\":" << rbt_orbbec::json_quote(motion.time_basis)
           << ",\"selection_policy\":\"nearest\""
           << ",\"timestamp_ns\":" << motion.lookup.requested_timestamp.ns
           << ",\"clock_kind\":"
           << rbt_orbbec::json_quote(clock_kind_name(motion.lookup.requested_timestamp.clock.kind()))
           << ",\"clock_source\":"
           << rbt_orbbec::json_quote(
                  motion.lookup.requested_timestamp.clock.source_id().to_hex())
           << ",\"clock_generation\":"
           << motion.lookup.requested_timestamp.clock.generation()
           << ",\"pitch_sample_timestamp_ns\":" << motion.pitch_sample_timestamp.ns
           << ",\"pitch_sample_clock_kind\":"
           << rbt_orbbec::json_quote(clock_kind_name(motion.pitch_sample_timestamp.clock.kind()))
           << ",\"pitch_sample_clock_source\":"
           << rbt_orbbec::json_quote(motion.pitch_sample_timestamp.clock.source_id().to_hex())
           << ",\"pitch_sample_clock_generation\":"
           << motion.pitch_sample_timestamp.clock.generation()
           << ",\"pitch_sample_offset_ns\":" << motion.pitch_sample_offset_ns
           << ",\"pitch_sample_offset_abs_ns\":" << motion.pitch_sample_offset_abs_ns
           << ",\"timing_uncertainty_ns\":";
    if (motion.lookup.timing_uncertainty_ns.has_value()) {
        output << *motion.lookup.timing_uncertainty_ns;
    } else {
        output << "null";
    }
    output << ",\"camera_in_world\":";
    append_pose(output, motion.camera_in_world);
    output << ",\"transform_source_count\":" << motion.lookup.sources.size() << '}';
}

void append_joint_prior_json(std::ostringstream& output, const JointPriorReport& report) {
    output << ",\"joint_priors\":{\"enabled\":true,\"status\":"
           << rbt_orbbec::json_quote(report.status)
           << ",\"time_basis\":" << rbt_orbbec::json_quote(report.time_basis)
           << ",\"controller_id\":" << rbt_orbbec::json_quote(report.controller_id)
           << ",\"driver_session\":" << rbt_orbbec::json_quote(report.driver_session)
           << ",\"sequence\":" << report.sequence << ",\"age_ns\":";
    if (report.age_ns.has_value()) {
        output << *report.age_ns;
    } else {
        output << "null";
    }
    output << ",\"capture_time_associated\":"
           << (report.capture_time_associated ? "true" : "false")
           << ",\"prior_count\":" << report.priors.size() << ",\"priors\":[";
    for (std::size_t index = 0U; index < report.priors.size(); ++index) {
        const auto& prior = report.priors[index];
        if (!valid_frame_prefix(prior.arm) || prior.joint.empty()
            || !std::isfinite(prior.angle_rad)
            || !std::isfinite(prior.standard_deviation_rad)
            || prior.standard_deviation_rad <= 0.0) {
            invalid("tracking estimate contains an invalid joint prior");
        }
        if (index != 0U) {
            output << ',';
        }
        output << "{\"arm\":" << rbt_orbbec::json_quote(prior.arm)
               << ",\"joint\":" << rbt_orbbec::json_quote(prior.joint)
               << ",\"angle_rad\":" << prior.angle_rad
               << ",\"standard_deviation_rad\":" << prior.standard_deviation_rad << '}';
    }
    output << "],\"rejected_mappings\":[";
    for (std::size_t index = 0U; index < report.rejected_mappings.size(); ++index) {
        if (index != 0U) {
            output << ',';
        }
        output << rbt_orbbec::json_quote(report.rejected_mappings[index]);
    }
    output << "]}";
}

void append_mount_refinement_json(std::ostringstream& output, const TrackingEstimate& estimate) {
    if (estimate.mount_refinement.empty()) {
        if (estimate.joints_locked_to_feedback) {
            invalid("joints_locked_to_feedback requires mount_refinement diagnostics");
        }
        return;
    }
    if (estimate.joints_locked_to_feedback == estimate.encoder_reference_links) {
        invalid("mount_refinement joints_locked_to_feedback must agree with encoder_reference_links");
    }
    std::set<std::string> link_arms;
    for (const auto& link : estimate.links) {
        link_arms.insert(link.arm);
    }
    std::set<std::string> joint_arms;
    for (const auto& entry : estimate.joint_angles) {
        joint_arms.insert(entry.first);
    }
    output << ",\"mount_refinement\":{\"enabled\":true,\"joints_locked_to_feedback\":"
           << (estimate.joints_locked_to_feedback ? "true" : "false") << ",\"arms\":[";
    std::set<std::string> arms;
    for (std::size_t index = 0U; index < estimate.mount_refinement.size(); ++index) {
        const auto& refinement = estimate.mount_refinement[index];
        if (!valid_frame_prefix(refinement.arm) || !arms.insert(refinement.arm).second
            || !std::isfinite(refinement.translation_correction_m)
            || refinement.translation_correction_m < 0.0
            || !std::isfinite(refinement.rotation_correction_rad)
            || refinement.rotation_correction_rad < 0.0) {
            invalid("tracking estimate contains invalid mount_refinement diagnostics");
        }
        if (!is_rigid_pose(refinement.nominal_base_in_camera)
            || !is_rigid_pose(refinement.refined_base_in_camera)) {
            invalid("mount_refinement poses must be finite proper rigid transforms");
        }
        const double translation = (refinement.refined_base_in_camera.translation().cast<double>()
                                    - refinement.nominal_base_in_camera.translation().cast<double>()).norm();
        const Eigen::Matrix3d rotation = refinement.refined_base_in_camera.linear().cast<double>()
                                         * refinement.nominal_base_in_camera.linear().cast<double>().transpose();
        const double sine = 0.5 * Eigen::Vector3d {
            rotation(2, 1) - rotation(1, 2), rotation(0, 2) - rotation(2, 0),
            rotation(1, 0) - rotation(0, 1) }.norm();
        const double angle = std::atan2(sine, std::clamp((rotation.trace() - 1.0) / 2.0, -1.0, 1.0));
        if (std::abs(refinement.translation_correction_m - translation) > 1.0e-4
            || std::abs(refinement.rotation_correction_rad - angle) > 1.0e-4) {
            invalid("mount_refinement correction magnitudes disagree with the reported poses");
        }
        if (index != 0U) {
            output << ',';
        }
        output << "{\"arm\":" << rbt_orbbec::json_quote(refinement.arm)
               << ",\"nominal_base_in_camera\":";
        append_pose(output, refinement.nominal_base_in_camera);
        output << ",\"refined_base_in_camera\":";
        append_pose(output, refinement.refined_base_in_camera);
        output << ",\"translation_correction_m\":" << refinement.translation_correction_m
               << ",\"rotation_correction_rad\":" << refinement.rotation_correction_rad << '}';
    }
    // The estimate has no RobotModel/root name: require agreement with both
    // independent arm identities instead of assuming a link named base_link.
    if (arms != link_arms || arms != joint_arms) {
        invalid("mount_refinement arms must match link and joint-angle arms");
    }
    output << "]}";
}

} // namespace

TrackingConfig load_tracking_config(const std::filesystem::path& path) {
    if (path.empty()) {
        invalid("tracking configuration path must be set");
    }
    auto storage = open_json(path, "tracking configuration JSON");
    const auto root = storage.root();
    validate_keys(root,
                  { "schema_version", "robot_model", "initialization", "cache_directory",
                    "model_divides", "model_points", "model_image_size",
                    "correspondence_iterations", "update_iterations", "modalities",
                    "occlusion", "metadata", "initialization_flow", "visualization",
                    "constraints", "joint_priors", "camera_motion", "secondary_camera",
                    "mount_refinement", "encoder_reference_links", "visual_link_weights",
                    "tf_nominal_initialization" },
                  "tracking configuration JSON");
    if (read_integer(required(root, "schema_version", "schema_version"), "schema_version") != 1) {
        invalid("unsupported tracking configuration schema_version");
    }

    const auto robot_path = resolve_relative_path(
        path, required(root, "robot_model", "robot_model"), "robot_model");
    const auto initialization_path = resolve_relative_path(
        path, required(root, "initialization", "initialization"), "initialization");
    EngineOptions engine;
    engine.cache_directory = resolve_relative_path(
        path, required(root, "cache_directory", "cache_directory"), "cache_directory");

    const auto option = [&root](const char* key, int& target, int minimum, int maximum) {
        if (has_key(root, key)) {
            const auto node = root[key];
            target = read_integer(node, key);
            if (target < minimum || target > maximum) {
                invalid(std::string(key) + " is outside the supported range");
            }
        }
    };
    option("model_divides", engine.model_divides, 0, 4);
    option("model_points", engine.model_points, 32, 1000);
    option("model_image_size", engine.model_image_size, 128, 2048);
    option("correspondence_iterations", engine.correspondence_iterations, 1, 20);
    option("update_iterations", engine.update_iterations, 1, 20);
    read_nested_options(path, root, engine);
    if (has_key(root, "encoder_reference_links")) {
        engine.encoder_reference_links = read_boolean(root["encoder_reference_links"],
                                                      "encoder_reference_links");
    }

    auto model = load_robot_model(robot_path);
    engine.visual_link_weights = read_visual_link_weights(root, model);
    auto initialization = read_initialization(initialization_path, model);
    std::vector<ArmSeed> tf_nominal_arms;
    if (has_key(root, "tf_nominal_initialization")) {
        if (!engine.encoder_reference_links) {
            invalid("tf_nominal_initialization requires encoder_reference_links");
        }
        const auto nominal_path = resolve_relative_path(
            path, root["tf_nominal_initialization"], "tf_nominal_initialization");
        auto nominal = read_initialization(nominal_path, model);
        if (nominal.coordinate_frame != initialization.coordinate_frame
            || nominal.serial != initialization.serial
            || nominal.arms.size() != initialization.arms.size()
            || !std::all_of(initialization.arms.begin(), initialization.arms.end(),
                [&nominal](const ArmSeed& arm) {
                    return std::any_of(nominal.arms.begin(), nominal.arms.end(),
                        [&arm](const ArmSeed& anchor) { return anchor.name == arm.name; });
                })) {
            invalid("TF nominal initialization must match the tracking camera and arm identities");
        }
        tf_nominal_arms = std::move(nominal.arms);
    }
    engine.constraints = read_constraints(path, root, model, initialization.arms);
    auto joint_prior = read_joint_prior_config(root);
    validate_joint_prior_config(joint_prior, model, initialization.arms);
    auto camera_motion = read_camera_motion_config(root, initialization.coordinate_frame,
                                                   initialization.arms);
    if (engine.encoder_reference_links && (engine.use_static_detector || engine.refine_initialization)) {
        invalid("encoder_reference_links requires measured initialization without a detector or refiner");
    }
    if (engine.encoder_reference_links || engine.mount_refinement.enabled) {
        const std::string mode = engine.encoder_reference_links
                                     ? "encoder_reference_links" : "mount_refinement";
        if (!joint_prior.enabled || !joint_prior.required
            || !joint_prior.initialize_from_measurement) {
            invalid(mode + " requires enabled, required joint_priors with "
                    "initialize_from_measurement enabled");
        }
        if (engine.mount_refinement.enabled
            && (camera_motion.enabled || engine.use_static_detector || engine.refine_initialization)) {
            invalid("mount_refinement requires a static camera and disabled static_detector/refine");
        }
        std::set<std::pair<std::string, std::string>> mapped;
        for (const auto& mapping : joint_prior.mappings) {
            mapped.emplace(mapping.arm, mapping.joint);
        }
        for (const auto& arm : initialization.arms) {
            if (engine.mount_refinement.enabled && arm.fixed_base) {
                invalid("mount_refinement requires every arm seed fixed_base to be false");
            }
            for (const auto& joint : model.joints) {
                if (joint.type == "revolute" && mapped.count({ arm.name, joint.name }) == 0U) {
                    invalid(mode + " requires joint_priors for every arm's revolute joints");
                }
            }
        }
    }
    auto secondary_camera = read_secondary_camera_config(
        root, initialization.coordinate_frame, initialization.serial);
    return TrackingConfig { std::move(model), std::move(initialization.arms), std::move(engine),
                            std::move(initialization.coordinate_frame),
                            std::move(initialization.serial), initialization_path,
                            std::move(joint_prior), std::move(camera_motion),
                            std::move(secondary_camera), std::move(tf_nominal_arms) };
}

TrackingInitialization load_tracking_initialization(const std::filesystem::path& path,
                                                    const RobotModel& model) {
    if (path.empty()) {
        invalid("initialization path must be set");
    }
    auto initialization = read_initialization(path, model);
    return TrackingInitialization { std::move(initialization.arms),
                                    std::move(initialization.coordinate_frame),
                                    std::move(initialization.serial) };
}

std::string tracking_json(const TrackingEstimate& estimate,
                          const RectifiedPair& pair,
                          std::uint64_t frame_sequence,
                          std::uint64_t dropped_frames,
                          const CameraMotionSelection* motion,
                          const RectifiedPair* secondary_pair,
                          const SecondaryCameraConfig* secondary_config) {
    if ((secondary_pair == nullptr) != (secondary_config == nullptr)) {
        invalid("dual-camera tracking output requires both secondary pair and configuration");
    }
    if (!pair.timing.capture.has_value()) {
        invalid("tracking output requires a depth capture timestamp");
    }
    const auto& depth_capture = *pair.timing.capture;
    const auto& color_capture = pair.color_capture;
    if (!depth_capture.clock.is_well_formed() || !color_capture.clock.is_well_formed()
        || !supported_capture_clock(depth_capture.clock)
        || depth_capture.clock != color_capture.clock
        || pair.skew_ns != timestamp_distance(depth_capture.ns, color_capture.ns)) {
        invalid("tracking output pair has inconsistent capture clocks or timestamps");
    }
    try {
        (void)rbt::tf::FrameId(pair.frame_id);
    } catch (const rbt::tf::TfError&) {
        invalid("tracking output frame_id must be a valid RBT frame id");
    }
    if (estimate.links.empty() || estimate.links.size() > kMaximumPoseLinks) {
        invalid("tracking output must contain between one and 64 link poses");
    }
    if (estimate.modalities.empty()) {
        invalid("tracking output must contain at least one modality");
    }
    std::set<std::string> modalities;
    for (const auto& modality : estimate.modalities) {
        if (modality != "region" && modality != "depth" && modality != "texture") {
            invalid("tracking output contains an unknown modality");
        }
        if (!modalities.insert(modality).second) {
            invalid("tracking output contains a duplicate modality");
        }
    }
    const bool legacy_modalities = estimate.modalities
                                   == std::vector<std::string> { "region", "depth" };

    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    output << "{\"schema_version\":1,\"event\":\"estimate\",\"algorithm\":"
           << rbt_orbbec::json_quote(legacy_modalities ? "m3t_region_depth_articulated"
                                                       : "m3t_articulated")
           << ",\"modalities\":[";
    for (std::size_t index = 0U; index < estimate.modalities.size(); ++index) {
        if (index != 0U) {
            output << ',';
        }
        output << rbt_orbbec::json_quote(estimate.modalities[index]);
    }
    output << "],\"frame_sequence\":"
           << frame_sequence << ",\"dropped_frames\":" << dropped_frames
           << ",\"frame_id\":" << rbt_orbbec::json_quote(pair.frame_id)
           << ",\"capture_ns\":" << depth_capture.ns
           << ",\"color_capture_ns\":" << color_capture.ns
           << ",\"pair_skew_ns\":" << pair.skew_ns
           << ",\"capture_clock_kind\":"
           << rbt_orbbec::json_quote(clock_kind_name(depth_capture.clock.kind()))
           << ",\"capture_clock_source\":"
           << rbt_orbbec::json_quote(depth_capture.clock.source_id().to_hex())
           << ",\"capture_clock_generation\":" << depth_capture.clock.generation()
           << ",\"uncertainty_ns\":";
    if (pair.timing.capture_uncertainty_ns.has_value()) {
        output << *pair.timing.capture_uncertainty_ns;
    } else {
        output << "null";
    }
    if (motion != nullptr) {
        append_camera_motion_json(output, pair, *motion);
    }
    if (secondary_pair != nullptr) {
        append_secondary_camera_json(output, pair, *secondary_pair, *secondary_config);
    }

    output << ",\"links\":[";
    std::set<std::string> frame_ids;
    for (std::size_t index = 0U; index < estimate.links.size(); ++index) {
        const auto& link = estimate.links[index];
        if (!valid_frame_prefix(link.arm) || link.link.empty()) {
            invalid("tracking estimate contains an invalid arm or link name");
        }
        const auto link_frame = link.arm + "/" + link.link;
        try {
            (void)rbt::tf::FrameId(link_frame);
        } catch (const rbt::tf::TfError&) {
            invalid("tracking estimate contains an invalid link frame id");
        }
        if (!frame_ids.insert(link_frame).second) {
            invalid("tracking estimate contains a duplicate link frame id");
        }
        if (index != 0U) {
            output << ',';
        }
        output << "{\"arm\":" << rbt_orbbec::json_quote(link.arm)
               << ",\"link\":" << rbt_orbbec::json_quote(link.link)
               << ",\"frame_id\":" << rbt_orbbec::json_quote(link_frame)
               << ",\"pose\":";
        append_pose(output, link.in_camera);
        output << '}';
    }

    output << "],\"joint_angles_rad\":{";
    bool first_arm = true;
    for (const auto& [arm, angles] : estimate.joint_angles) {
        if (!valid_frame_prefix(arm)) {
            invalid("tracking estimate contains an invalid joint-angle arm name");
        }
        if (!first_arm) {
            output << ',';
        }
        first_arm = false;
        output << rbt_orbbec::json_quote(arm) << ":{";
        bool first_joint = true;
        for (const auto& [joint, angle] : angles) {
            if (joint.empty() || !std::isfinite(angle)) {
                invalid("tracking estimate contains an invalid joint angle");
            }
            if (!first_joint) {
                output << ',';
            }
            first_joint = false;
            output << rbt_orbbec::json_quote(joint) << ':' << angle;
        }
        output << '}';
    }
    output << '}';
    if (estimate.joint_prior.enabled) {
        append_joint_prior_json(output, estimate.joint_prior);
    }
    if (estimate.encoder_reference_links) {
        output << ",\"encoder_reference_links\":true";
    }
    append_mount_refinement_json(output, estimate);
    append_visual_link_weights_json(output, estimate.visual_link_weights);
    output << '}';
    return output.str();
}

} // namespace rbt_m3t
