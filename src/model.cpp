#include <rbt_m3t/model.hpp>
#include <rbt_m3t/weighted_modality.hpp>

#include <m3t/body.h>
#include <m3t/basic_depth_renderer.h>
#include <m3t/constraint.h>
#include <m3t/depth_modality.h>
#include <m3t/depth_model.h>
#include <m3t/link.h>
#include <m3t/optimizer.h>
#include <m3t/refiner.h>
#include <m3t/region_modality.h>
#include <m3t/region_model.h>
#include <m3t/soft_constraint.h>
#include <m3t/static_detector.h>
#include <m3t/texture_modality.h>
#include <m3t/silhouette_renderer.h>
#include <m3t/tracker.h>

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rbt_m3t {

namespace detail {

double unwrap_and_clamp_joint_angle(double wrapped, double previous,
                                    double lower, double upper) {
    constexpr double pi = 3.14159265358979323846;
    const auto turns = std::round((previous - wrapped) / (2.0 * pi));
    return std::clamp(wrapped + turns * (2.0 * pi), lower, upper);
}

} // namespace detail

namespace {

constexpr std::size_t kMaximumLinks = 32U;
constexpr std::size_t kMaximumNameBytes = 256U;
constexpr float kRigidTolerance = 1.0e-4F;
constexpr float kAxisTolerance = 1.0e-5F;
// Allow for float pose roundoff around the native residual's discontinuity at pi.
constexpr double kMaximumEncoderCorrection = std::numbers::pi - kRigidTolerance;

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument(message);
}

[[noreturn]] void failed(const std::string& message) {
    throw std::runtime_error(message);
}

[[nodiscard]] bool finite_matrix(const Eigen::Matrix4f& matrix) noexcept {
    return matrix.array().isFinite().all();
}

[[nodiscard]] bool is_rigid_pose(const Eigen::Affine3f& pose) noexcept {
    const auto matrix = pose.matrix();
    if (!finite_matrix(matrix)) {
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

void validate_camera_pose(const Eigen::Affine3f& pose) {
    if (!is_rigid_pose(pose)) {
        invalid("M3T camera pose must be a finite proper rigid transform");
    }
}

void validate_name(const std::string& name, const char* description) {
    if (name.empty() || name.size() > kMaximumNameBytes) {
        invalid(std::string(description) + " must be non-empty and at most 256 bytes");
    }
}

[[nodiscard]] bool is_number(const cv::FileNode& node) noexcept {
    return node.isInt() || node.isReal();
}

[[nodiscard]] std::string read_string(const cv::FileNode& node,
                                      const char* description) {
    if (!node.isString()) {
        invalid(std::string(description) + " must be a string");
    }
    std::string value;
    node >> value;
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

[[nodiscard]] Eigen::Vector3f read_vector3(const cv::FileNode& node,
                                           const char* description) {
    if (!node.isSeq() || node.size() != 3U) {
        invalid(std::string(description) + " must contain exactly three numbers");
    }
    Eigen::Vector3f result;
    for (int index = 0; index < 3; ++index) {
        const auto value = read_number(node[index], description);
        if (std::abs(value) > std::numeric_limits<float>::max()) {
            invalid(std::string(description) + " exceeds the supported range");
        }
        result[index] = static_cast<float>(value);
    }
    return result;
}

[[nodiscard]] Eigen::Affine3f read_origin(const cv::FileNode& node) {
    if (!node.isSeq() || node.size() != 16U) {
        invalid("joint origin must contain exactly 16 row-major numbers");
    }
    Eigen::Affine3f result = Eigen::Affine3f::Identity();
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            const auto value = read_number(node[row * 4 + column], "joint origin");
            if (std::abs(value) > std::numeric_limits<float>::max()) {
                invalid("joint origin exceeds the supported range");
            }
            result.matrix()(row, column) = static_cast<float>(value);
        }
    }
    if (!is_rigid_pose(result)) {
        invalid("joint origin must be a finite proper rigid transform");
    }
    return result;
}

[[nodiscard]] std::filesystem::path resolve_mesh(
    const std::filesystem::path& json_path, const std::string& mesh) {
    if (mesh.empty()) {
        return {};
    }
    const std::filesystem::path relative { mesh };
    if (relative.is_absolute() || relative.has_root_name() || relative.has_root_directory()) {
        invalid("link mesh paths must be relative to robot.json");
    }
    for (const auto& component : relative) {
        if (component == "..") {
            invalid("link mesh paths may not escape the robot model directory");
        }
    }
    if (relative.extension() != ".obj") {
        invalid("link meshes must use relative OBJ paths");
    }
    const auto base = json_path.parent_path().empty() ? std::filesystem::current_path()
                                                      : json_path.parent_path();
    return std::filesystem::absolute(base / relative).lexically_normal();
}

struct ModelIndex {
    std::unordered_map<std::string, std::size_t> link_indices;
    std::vector<std::vector<std::size_t>> child_joint_indices;
    std::vector<std::size_t> revolute_joint_indices;
    std::size_t root_index {};
};

[[nodiscard]] ModelIndex validate_model(const RobotModel& model) {
    validate_name(model.name, "robot name");
    validate_name(model.root_link, "root link name");
    if (model.links.empty() || model.links.size() > kMaximumLinks) {
        invalid("robot model must contain between one and 32 links");
    }
    if (model.joints.size() + 1U != model.links.size()) {
        invalid("robot model must be one connected tree with one fewer joint than links");
    }

    ModelIndex index;
    index.child_joint_indices.resize(model.links.size());
    for (std::size_t i = 0; i < model.links.size(); ++i) {
        const auto& link = model.links[i];
        validate_name(link.name, "link name");
        if (!index.link_indices.emplace(link.name, i).second) {
            invalid("robot model contains duplicate link names");
        }
        if (!link.mesh.empty()) {
            if (link.mesh.is_absolute()) {
                // Programmatically constructed models may already hold an absolute path.
                if (link.mesh.extension() != ".obj") {
                    invalid("link meshes must use OBJ files");
                }
            } else {
                if (link.mesh.extension() != ".obj") {
                    invalid("link meshes must use OBJ files");
                }
                for (const auto& component : link.mesh) {
                    if (component == "..") {
                        invalid("link mesh paths may not contain parent directory components");
                    }
                }
            }
        }
    }
    const auto root = index.link_indices.find(model.root_link);
    if (root == index.link_indices.end()) {
        invalid("robot root link is not present in the link list");
    }
    index.root_index = root->second;

    std::set<std::string> joint_names;
    std::vector<int> parent_counts(model.links.size(), 0);
    for (std::size_t i = 0; i < model.joints.size(); ++i) {
        const auto& joint = model.joints[i];
        validate_name(joint.name, "joint name");
        if (!joint_names.insert(joint.name).second) {
            invalid("robot model contains duplicate joint names");
        }
        if (joint.type != "fixed" && joint.type != "revolute") {
            invalid("joint type must be fixed or revolute");
        }
        const auto parent = index.link_indices.find(joint.parent);
        const auto child = index.link_indices.find(joint.child);
        if (parent == index.link_indices.end() || child == index.link_indices.end()) {
            invalid("joint references a link that is not present in the model");
        }
        if (parent->second == child->second) {
            invalid("joint may not connect a link to itself");
        }
        if (!is_rigid_pose(joint.origin)) {
            invalid("joint origin must be a finite proper rigid transform");
        }
        if (!joint.axis.array().isFinite().all()) {
            invalid("joint axis must be finite");
        }
        if (!std::isfinite(joint.lower) || !std::isfinite(joint.upper)
            || std::abs(joint.lower) > std::numeric_limits<float>::max()
            || std::abs(joint.upper) > std::numeric_limits<float>::max()
            || joint.lower > joint.upper) {
            invalid("joint limits must be finite, float-representable, and ordered");
        }
        if (joint.type == "revolute") {
            if ((joint.axis - Eigen::Vector3f::UnitZ()).cwiseAbs().maxCoeff()
                > kAxisTolerance) {
                invalid("only +Z revolute joint axes are supported");
            }
            index.revolute_joint_indices.push_back(i);
        }
        ++parent_counts[child->second];
        if (parent_counts[child->second] > 1) {
            invalid("a link may have only one parent joint");
        }
        index.child_joint_indices[parent->second].push_back(i);
    }

    for (std::size_t i = 0; i < model.links.size(); ++i) {
        if (i == index.root_index) {
            if (parent_counts[i] != 0) {
                invalid("the declared root link may not have a parent joint");
            }
        } else if (parent_counts[i] != 1) {
            invalid("every non-root link must have exactly one parent joint");
        }
    }

    std::vector<bool> visited(model.links.size(), false);
    std::vector<std::size_t> stack { index.root_index };
    while (!stack.empty()) {
        const auto link_index = stack.back();
        stack.pop_back();
        if (visited[link_index]) {
            invalid("robot joint graph contains a cycle");
        }
        visited[link_index] = true;
        for (const auto joint_index : index.child_joint_indices[link_index]) {
            stack.push_back(index.link_indices.at(model.joints[joint_index].child));
        }
    }
    if (std::find(visited.begin(), visited.end(), false) != visited.end()) {
        invalid("robot joint graph is disconnected from its root");
    }

    return index;
}

[[nodiscard]] std::map<std::string, double> validate_seed(
    const RobotModel& model, const ModelIndex& index, const ArmSeed& seed) {
    validate_name(seed.name, "arm name");
    if (!is_rigid_pose(seed.base_in_camera)) {
        invalid("arm base pose must be a finite proper rigid transform");
    }
    if (seed.joint_angles.size() != index.revolute_joint_indices.size()) {
        invalid("arm seed must provide exactly one angle for every revolute joint");
    }
    for (const auto& [name, angle] : seed.joint_angles) {
        const auto found = std::find_if(model.joints.begin(), model.joints.end(),
                                        [&name](const ModelJoint& joint) {
                                            return joint.name == name;
                                        });
        if (found == model.joints.end() || found->type != "revolute") {
            invalid("arm seed contains an unknown or fixed joint angle");
        }
        if (!std::isfinite(angle) || angle < found->lower || angle > found->upper) {
            invalid("arm seed joint angles must be finite and within their limits");
        }
    }
    for (const auto joint_index : index.revolute_joint_indices) {
        const auto& joint = model.joints[joint_index];
        if (seed.joint_angles.find(joint.name) == seed.joint_angles.end()) {
            invalid("arm seed is missing a revolute joint angle");
        }
    }
    return seed.joint_angles;
}

[[nodiscard]] Eigen::Affine3f revolute_rotation(double angle) {
    Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
    rotation.rotate(Eigen::AngleAxisf(static_cast<float>(angle),
                                     Eigen::Vector3f::UnitZ()));
    return rotation;
}

[[nodiscard]] Eigen::Affine3f joint_transform(const ModelJoint& joint,
                                              double angle) {
    return joint.type == "revolute" ? joint.origin * revolute_rotation(angle)
                                     : joint.origin;
}

[[nodiscard]] std::vector<Eigen::Affine3f> calculate_poses(
    const RobotModel& model, const ModelIndex& index, const ArmSeed& seed) {
    const auto angles = validate_seed(model, index, seed);
    std::vector<Eigen::Affine3f> poses(model.links.size(), Eigen::Affine3f::Identity());
    poses[index.root_index] = seed.base_in_camera;
    std::vector<std::size_t> stack { index.root_index };
    while (!stack.empty()) {
        const auto parent_index = stack.back();
        stack.pop_back();
        for (const auto joint_index : index.child_joint_indices[parent_index]) {
            const auto& joint = model.joints[joint_index];
            const auto child_index = index.link_indices.at(joint.child);
            const double angle = joint.type == "revolute" ? angles.at(joint.name) : 0.0;
            poses[child_index] = poses[parent_index] * joint_transform(joint, angle);
            if (!is_rigid_pose(poses[child_index])) {
                invalid("forward kinematics produced a non-finite rigid pose");
            }
            stack.push_back(child_index);
        }
    }
    return poses;
}

[[nodiscard]] Eigen::Affine3f to_eigen_affine(const m3t::Transform3fA& pose) {
    Eigen::Affine3f result = Eigen::Affine3f::Identity();
    result.matrix() = pose.matrix();
    return result;
}

[[nodiscard]] m3t::Transform3fA to_m3t_transform(const Eigen::Affine3f& pose) {
    m3t::Transform3fA result = m3t::Transform3fA::Identity();
    result.matrix() = pose.matrix();
    return result;
}

[[nodiscard]] bool valid_rgbd_pair(const RectifiedPair& pair) noexcept {
    const auto& intrinsics = pair.intrinsics;
    return !pair.bgr.empty() && pair.bgr.type() == CV_8UC3
           && !pair.depth_u16.empty() && pair.depth_u16.type() == CV_16UC1
           && pair.bgr.size() == pair.depth_u16.size()
           && intrinsics.width == pair.bgr.cols && intrinsics.height == pair.bgr.rows
           && std::isfinite(intrinsics.fu) && intrinsics.fu > 0.0F
           && std::isfinite(intrinsics.fv) && intrinsics.fv > 0.0F
           && std::isfinite(intrinsics.ppu) && std::isfinite(intrinsics.ppv)
           && std::isfinite(pair.depth_scale) && pair.depth_scale > 0.0;
}

[[nodiscard]] double z_rotation_angle(const Eigen::Affine3f& pose) {
    return std::atan2(static_cast<double>(pose.linear()(1, 0)),
                      static_cast<double>(pose.linear()(0, 0)));
}

struct ValidatedJointPrior {
    std::size_t arm_index {};
    std::size_t joint_index {};
    float target_angle {};
    float standard_deviation {};
    double measured_angle {};
};

[[nodiscard]] std::vector<ValidatedJointPrior> validate_joint_priors(
    const RobotModel& model, const ModelIndex& index,
    const std::vector<ArmSeed>& seeds,
    const std::vector<JointAnglePrior>& priors) {
    std::map<std::string, std::size_t> arm_indices;
    for (std::size_t i = 0; i < seeds.size(); ++i) {
        (void)validate_seed(model, index, seeds[i]);
        if (!arm_indices.emplace(seeds[i].name, i).second) {
            invalid("arm names must be unique when applying joint priors");
        }
    }

    std::set<std::pair<std::string, std::string>> seen_joints;
    std::vector<ValidatedJointPrior> result;
    result.reserve(priors.size());
    for (const auto& prior : priors) {
        const auto arm = arm_indices.find(prior.arm);
        if (arm == arm_indices.end()) {
            invalid("joint prior references an unknown arm");
        }
        const auto joint = std::find_if(model.joints.begin(), model.joints.end(),
                                        [&prior](const ModelJoint& candidate) {
                                            return candidate.name == prior.joint;
                                        });
        if (joint == model.joints.end() || joint->type != "revolute") {
            invalid("joint prior references an unknown or non-revolute joint");
        }
        if (!seen_joints.emplace(prior.arm, prior.joint).second) {
            invalid("joint priors may contain at most one value per arm and joint");
        }
        if (!std::isfinite(prior.angle_rad) || prior.angle_rad < joint->lower
            || prior.angle_rad > joint->upper) {
            invalid("joint prior angles must be finite and within their joint limits");
        }
        const float target_angle = static_cast<float>(prior.angle_rad);
        if (!std::isfinite(target_angle)) {
            invalid("joint prior angle must be float-representable");
        }
        if (!std::isfinite(prior.standard_deviation_rad)
            || prior.standard_deviation_rad <= 0.0) {
            invalid("joint prior standard deviations must be finite and positive");
        }
        const float standard_deviation =
            static_cast<float>(prior.standard_deviation_rad);
        if (!std::isfinite(standard_deviation) || standard_deviation <= 0.0F) {
            invalid("joint prior standard deviation must be positive and float-representable");
        }
        const double represented_standard_deviation = standard_deviation;
        const double variance = represented_standard_deviation
                                * represented_standard_deviation;
        const double precision = 1.0 / variance;
        const double smallest_positive_float =
            std::numeric_limits<float>::denorm_min() > 0.0F
                ? std::numeric_limits<float>::denorm_min()
                : std::numeric_limits<float>::min();
        if (!std::isfinite(precision) || precision <= 0.0
            || precision < smallest_positive_float
            || precision > std::numeric_limits<float>::max()) {
            invalid("joint prior precision must be positive and float-representable");
        }
        const float float_precision = static_cast<float>(precision);
        if (!std::isfinite(float_precision) || float_precision <= 0.0F) {
            invalid("joint prior precision must be positive and float-representable");
        }
        const double maximum_weighted_residual =
            (joint->upper - joint->lower) * precision;
        if (!std::isfinite(maximum_weighted_residual)
            || maximum_weighted_residual > std::numeric_limits<float>::max()) {
            invalid("joint prior weighted residual exceeds the supported float range");
        }
        result.push_back(ValidatedJointPrior {
            arm->second,
            static_cast<std::size_t>(std::distance(model.joints.begin(), joint)),
            target_angle,
            standard_deviation,
            prior.angle_rad,
        });
    }
    return result;
}

[[nodiscard]] std::string cache_component(const std::string& value) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string encoded;
    encoded.reserve(value.size() * 2U);
    for (const unsigned char byte : value) {
        encoded.push_back(hex[byte >> 4U]);
        encoded.push_back(hex[byte & 0x0FU]);
    }
    return encoded;
}

[[nodiscard]] std::string object_name(const std::string& arm,
                                      const std::string& link,
                                      const std::string& component) {
    return "arm_" + cache_component(arm) + "_link_" + cache_component(link) + "_"
           + component;
}

[[nodiscard]] std::string view_component(const std::string& view_name,
                                         const char* component) {
    return view_name == "primary" ? component
                                   : view_name + "_" + component;
}

void validate_options(const EngineOptions& options) {
    if (options.encoder_reference_links) {
        if (options.use_static_detector || options.refine_initialization) {
            invalid("encoder reference links do not support static detection or initialization refinement");
        }
    }
    if (options.mount_refinement.enabled) {
        const auto& mount = options.mount_refinement;
        if (!std::isfinite(mount.translation_standard_deviation_m)
            || mount.translation_standard_deviation_m < 0.0001
            || mount.translation_standard_deviation_m > 0.2
            || !std::isfinite(mount.rotation_standard_deviation_rad)
            || mount.rotation_standard_deviation_rad < 0.0001
            || mount.rotation_standard_deviation_rad > 0.785398) {
            invalid("mount refinement sigmas must be within T [0.0001, 0.2] m and R [0.0001, 0.785398] rad");
        }
        if (options.use_static_detector || options.refine_initialization) {
            invalid("mount refinement does not support static detection or initialization refinement");
        }
    }
    if (options.cache_directory.empty()) {
        invalid("M3T model cache directory must be set");
    }
    if (!options.use_region && !options.use_depth && !options.use_texture) {
        invalid("M3T requires at least one tracking modality");
    }
    if (options.model_divides < 0 || options.model_divides > 5
        || options.model_points < 1 || options.model_points > 10'000
        || options.model_image_size < 64 || options.model_image_size > 4096
        || options.occlusion_image_size < 64 || options.occlusion_image_size > 4096) {
        invalid("M3T model options are outside their supported bounds");
    }
    if (options.correspondence_iterations < 1 || options.correspondence_iterations > 20
        || options.update_iterations < 1 || options.update_iterations > 20) {
        invalid("M3T iteration counts must be between one and 20");
    }
}

[[nodiscard]] bool locks_joints_to_feedback(const EngineOptions& options) {
    return options.mount_refinement.enabled && !options.encoder_reference_links;
}

[[nodiscard]] int checked_visualization_save_index(std::uint64_t index) {
    if (index > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        invalid("M3T visualization save index exceeds the supported integer range");
    }
    return static_cast<int>(index);
}

void validate_depth_metadata(const EngineOptions& options) {
    if (!options.use_depth || options.depth_metadata.empty()) {
        return;
    }

    cv::FileStorage storage;
    try {
        if (!storage.open(options.depth_metadata.string(), cv::FileStorage::READ)) {
            invalid("could not open depth modality metadata: "
                    + options.depth_metadata.string());
        }
    } catch (const cv::Exception& error) {
        invalid("could not parse depth modality metadata: "
                + std::string(error.what()));
    }

    const auto measure_occlusions = storage["measure_occlusions"];
    if (!measure_occlusions.empty()) {
        if (!measure_occlusions.isInt()) {
            invalid("depth metadata measure_occlusions must be integer 0 or 1");
        }
        int configured = -1;
        measure_occlusions >> configured;
        if (configured != 0 && configured != 1) {
            invalid("depth metadata measure_occlusions must be integer 0 or 1");
        }
        if ((configured != 0) != options.measured_occlusions) {
            invalid("depth metadata measure_occlusions conflicts with the "
                    "EngineOptions measured_occlusions flag");
        }
    }
    storage.release();
}

} // namespace

RobotModel load_robot_model(const std::filesystem::path& path) {
    if (path.empty()) {
        invalid("robot JSON path must be set");
    }
    cv::FileStorage storage;
    try {
        if (!storage.open(path.string(), cv::FileStorage::READ)) {
            invalid("could not open robot JSON: " + path.string());
        }
    } catch (const cv::Exception& error) {
        invalid("could not parse robot JSON: " + std::string(error.what()));
    }

    RobotModel model;
    try {
        const auto schema_version = read_integer(storage["schema_version"], "schema_version");
        if (schema_version != 1) {
            invalid("unsupported robot JSON schema_version");
        }
        model.name = read_string(storage["name"], "robot name");
        model.root_link = read_string(storage["root_link"], "root_link");
        if (read_string(storage["units"], "units") != "metres_radians") {
            invalid("robot JSON units must be metres_radians");
        }

        const auto links_node = storage["links"];
        if (!links_node.isSeq() || links_node.empty() || links_node.size() > kMaximumLinks) {
            invalid("robot JSON links must contain between one and 32 entries");
        }
        model.links.reserve(links_node.size());
        for (const auto& node : links_node) {
            ModelLink link;
            link.name = read_string(node["name"], "link name");
            const auto mesh = read_string(node["mesh"], "link mesh");
            const int visuals = read_integer(node["visuals"], "link visuals count");
            if (visuals < 0) {
                invalid("link visuals count must be non-negative");
            }
            link.mesh = resolve_mesh(path, mesh);
            model.links.push_back(std::move(link));
        }

        const auto joints_node = storage["joints"];
        if (!joints_node.isSeq() || joints_node.size() + 1U != model.links.size()) {
            invalid("robot JSON joints must have one fewer entry than links");
        }
        model.joints.reserve(joints_node.size());
        for (const auto& node : joints_node) {
            ModelJoint joint;
            joint.name = read_string(node["name"], "joint name");
            joint.type = read_string(node["type"], "joint type");
            joint.parent = read_string(node["parent"], "joint parent");
            joint.child = read_string(node["child"], "joint child");
            joint.origin = read_origin(node["origin"]);
            joint.axis = read_vector3(node["axis"], "joint axis");
            joint.lower = read_number(node["lower"], "joint lower limit");
            joint.upper = read_number(node["upper"], "joint upper limit");
            model.joints.push_back(std::move(joint));
        }
    } catch (const cv::Exception& error) {
        storage.release();
        invalid("could not read robot JSON: " + std::string(error.what()));
    }
    storage.release();
    (void)validate_model(model);
    return model;
}

std::vector<LinkEstimate> forward_kinematics(const RobotModel& model,
                                             const ArmSeed& seed) {
    const auto index = validate_model(model);
    const auto poses = calculate_poses(model, index, seed);
    std::vector<LinkEstimate> estimates;
    estimates.reserve(model.links.size());
    for (std::size_t i = 0; i < model.links.size(); ++i) {
        estimates.push_back(LinkEstimate { seed.name, model.links[i].name, poses[i] });
    }
    return estimates;
}

std::vector<ArmSeed> seeds_with_joint_priors(
    const RobotModel& model, const std::vector<ArmSeed>& seeds,
    const std::vector<JointAnglePrior>& priors) {
    const auto index = validate_model(model);
    const auto validated_priors = validate_joint_priors(model, index, seeds, priors);
    auto result = seeds;
    for (std::size_t i = 0; i < priors.size(); ++i) {
        result[validated_priors[i].arm_index].joint_angles[priors[i].joint] =
            priors[i].angle_rad;
    }
    for (const auto& seed : result) {
        (void)calculate_poses(model, index, seed);
    }
    return result;
}

class TrackerEngine::Impl {
  public:
    struct ViewState {
        std::string name;
        std::shared_ptr<CameraImages> images;
        std::shared_ptr<RbtColorCamera> color_camera;
        std::shared_ptr<RbtDepthCamera> depth_camera;
        std::shared_ptr<m3t::FocusedBasicDepthRenderer> depth_renderer;
        std::shared_ptr<m3t::FocusedSilhouetteRenderer> region_silhouette_renderer;
        std::shared_ptr<m3t::FocusedSilhouetteRenderer> body_silhouette_renderer;
    };

    struct JointState {
        std::size_t joint_index {};
        std::shared_ptr<m3t::Link> link;
        double angle {};
        std::shared_ptr<m3t::Link> encoder_reference {};
        std::shared_ptr<m3t::SoftConstraint> encoder_constraint {};
        double encoder_angle {};
    };

    struct ArmState {
        ArmSeed seed;
        std::shared_ptr<m3t::Optimizer> optimizer;
        std::shared_ptr<m3t::StaticDetector> static_detector;
        std::shared_ptr<m3t::Link> root; // Physical base; never the internal CAD anchor.
        std::shared_ptr<m3t::Link> cad_anchor;
        std::unordered_map<std::string, std::shared_ptr<m3t::Link>> links;
        std::vector<JointState> joints;
    };

    Impl(RobotModel robot, std::vector<ArmSeed> arm_seeds, EngineOptions engine_options,
         std::shared_ptr<CameraImages> camera_images,
         std::shared_ptr<CameraImages> secondary_camera_images)
        : model{std::move(robot)}, seeds{std::move(arm_seeds)}, options{std::move(engine_options)},
          owner_thread{std::this_thread::get_id()} {
        index = validate_model(model);
        validate_options(options);
        for (const auto& [link_name, weight] : options.visual_link_weights) {
            const auto link = index.link_indices.find(link_name);
            if (link == index.link_indices.end() || model.links[link->second].mesh.empty()) {
                invalid("visual link weight requires an existing link with a mesh: " + link_name);
            }
            if (!std::isfinite(weight) || weight < 0.01 || weight > 100.0) {
                invalid("visual link weight must be finite and within [0.01, 100]: " + link_name);
            }
        }
        validate_depth_metadata(options);
        if (seeds.empty() || seeds.size() > 2U) {
            invalid("M3T articulated tracking requires one or two arm seeds");
        }
        std::set<std::string> arm_names;
        for (const auto& seed : seeds) {
            (void)validate_seed(model, index, seed);
            if (options.mount_refinement.enabled && seed.fixed_base) {
                invalid("mount refinement requires fixed_base=false for every arm");
            }
            if (!arm_names.insert(seed.name).second) {
                invalid("arm names must be unique");
            }
        }
        if (options.constraints.size() > 64U) {
            invalid("M3T supports at most 64 extra constraints");
        }
        std::set<std::string> constraint_names;
        for (const auto& constraint : options.constraints) {
            validate_name(constraint.name, "constraint name");
            validate_name(constraint.arm, "constraint arm name");
            validate_name(constraint.link1, "constraint link1 name");
            validate_name(constraint.link2, "constraint link2 name");
            if (!constraint_names.insert(constraint.name).second) {
                invalid("constraint names must be unique");
            }
            if (arm_names.find(constraint.arm) == arm_names.end()) {
                invalid("constraint references an unknown arm");
            }
            if (constraint.link1 == constraint.link2
                || index.link_indices.find(constraint.link1) == index.link_indices.end()
                || index.link_indices.find(constraint.link2) == index.link_indices.end()) {
                invalid("constraint links must be different existing model links");
            }
            std::error_code error;
            if (constraint.metadata.empty()
                || !std::filesystem::is_regular_file(constraint.metadata, error)
                || error) {
                invalid("constraint metadata must be a regular file");
            }
        }
        if (!camera_images) {
            invalid("M3T camera images must be provided");
        }
        if (secondary_camera_images && secondary_camera_images.get() == camera_images.get()) {
            invalid("M3T primary and secondary camera images must be independent objects");
        }
        ViewState primary_view;
        primary_view.name = "primary";
        primary_view.images = std::move(camera_images);
        views.push_back(std::move(primary_view));
        if (secondary_camera_images) {
            ViewState secondary_view;
            secondary_view.name = "secondary";
            secondary_view.images = std::move(secondary_camera_images);
            views.push_back(std::move(secondary_view));
        }
        tracker = std::make_shared<m3t::Tracker>(
            "rbt_m3t_dual_arm", options.correspondence_iterations,
            options.update_iterations, false, false, std::chrono::milliseconds { 33 }, 1, 1);
        for (auto& view : views) {
            const auto view_prefix = view.name == "primary" ? std::string{} : view.name + "_";
            view.color_camera = std::make_shared<RbtColorCamera>(
                "rbt_m3t_" + view_prefix + "color", view.images);
            view.depth_camera = std::make_shared<RbtDepthCamera>(
                "rbt_m3t_" + view_prefix + "depth", view.images);
        }
    }

    void ensure_owner_thread() const {
        if (std::this_thread::get_id() != owner_thread) {
            throw std::logic_error("M3T engine methods must stay on their construction thread");
        }
    }

    void validate_views(const char* operation) const {
        for (const auto& view : views) {
            validate_camera_pose(view.images->camera_in_world);
            if (!view.images->current || !valid_rgbd_pair(*view.images->current)) {
                invalid(std::string("M3T ") + operation
                        + " requires a valid caller-selected RGB-D pair for the "
                        + view.name + " view");
            }
        }
    }

    void prepare_renderers() {
        const bool use_depth_renderer = options.modeled_occlusions;
        const bool use_region_renderer = options.region_checking && options.use_region;
        const bool use_body_renderer = options.use_texture
                                       || (options.region_checking && options.use_depth);
        if (!use_depth_renderer && !use_region_renderer && !use_body_renderer) {
            return;
        }

        renderer_geometry = std::make_shared<m3t::RendererGeometry>("rbt_m3t_geometry");
        for (auto& view : views) {
            const auto name_suffix = view.name == "primary" ? std::string{}
                                                             : "_" + view.name;
            if (use_depth_renderer) {
                view.depth_renderer = std::make_shared<m3t::FocusedBasicDepthRenderer>(
                    "rbt_m3t_modeled_occlusion" + name_suffix, renderer_geometry,
                    view.depth_camera, options.occlusion_image_size);
            }
            if (use_region_renderer) {
                view.region_silhouette_renderer =
                    std::make_shared<m3t::FocusedSilhouetteRenderer>(
                        "rbt_m3t_region_silhouette" + name_suffix, renderer_geometry,
                        view.color_camera, m3t::IDType::REGION,
                        options.occlusion_image_size);
            }
            if (use_body_renderer) {
                view.body_silhouette_renderer =
                    std::make_shared<m3t::FocusedSilhouetteRenderer>(
                        "rbt_m3t_body_silhouette" + name_suffix, renderer_geometry,
                        view.color_camera, m3t::IDType::BODY,
                        options.occlusion_image_size);
            }
        }
    }

    void add_visual_body(const std::shared_ptr<m3t::Body>& body) {
        if (next_visual_body_id > 255U) {
            failed("M3T supports at most 255 visual body identifiers");
        }
        const auto id = static_cast<uchar>(next_visual_body_id++);
        body->set_body_id(id);
        body->set_region_id(id);
        target_bodies.push_back(body);

        if (renderer_geometry) {
            if (!renderer_geometry->AddBody(body)) {
                failed("could not add an M3T body to shared renderer geometry");
            }
            for (const auto& view : views) {
                if (view.depth_renderer && !view.depth_renderer->AddReferencedBody(body)) {
                    failed("could not add a body to a modeled-occlusion renderer");
                }
                if (view.region_silhouette_renderer
                    && !view.region_silhouette_renderer->AddReferencedBody(body)) {
                    failed("could not add a body to a region silhouette renderer");
                }
                if (view.body_silhouette_renderer
                    && !view.body_silhouette_renderer->AddReferencedBody(body)) {
                    failed("could not add a body to a body silhouette renderer");
                }
            }
        }
    }

    void build_arm(const ArmSeed& seed, std::size_t arm_number) {
        const auto camera_poses = calculate_poses(model, index, seed);
        std::vector<Eigen::Affine3f> initial_poses;
        initial_poses.reserve(camera_poses.size());
        for (const auto& pose : camera_poses) {
            const auto world_pose = views.front().images->camera_in_world * pose;
            if (!is_rigid_pose(world_pose)) {
                invalid("camera-frame seed produced a non-finite rigid world pose");
            }
            initial_poses.push_back(world_pose);
        }
        ArmState arm;
        arm.seed = seed;
        const auto arm_prefix = "arm" + std::to_string(arm_number) + "_" + cache_component(seed.name);
        const auto arm_cache = options.cache_directory / arm_prefix;
        std::filesystem::create_directories(arm_cache);

        std::function<std::shared_ptr<m3t::Link>(std::size_t, std::optional<std::size_t>)> make_link;
        make_link = [&](std::size_t link_index, std::optional<std::size_t> parent_joint_index) {
            const auto& link_model = model.links[link_index];
            const auto* parent_joint = parent_joint_index
                                           ? &model.joints[*parent_joint_index] : nullptr;
            const bool revolute = parent_joint && parent_joint->type == "revolute";
            const bool has_encoder_reference = options.encoder_reference_links && revolute;
            const double seed_angle = revolute ? seed.joint_angles.at(parent_joint->name) : 0.0;
            const auto joint_pose = parent_joint ? joint_transform(*parent_joint, seed_angle)
                                                 : Eigen::Affine3f::Identity();
            std::array<bool, 6> free_directions { false, false, false, false, false, false };
            if (revolute) {
                free_directions[2] = !locks_joints_to_feedback(options);
            } else if (!parent_joint && !seed.fixed_base) {
                free_directions = { true, true, true, true, true, true };
            }

            std::shared_ptr<m3t::Body> body;
            std::vector<std::shared_ptr<m3t::RegionModality>> region_modalities;
            std::vector<std::shared_ptr<m3t::DepthModality>> depth_modalities;
            std::vector<std::shared_ptr<m3t::TextureModality>> texture_modalities;
            if (!link_model.mesh.empty()) {
                const auto name = object_name(seed.name, link_model.name, "body");
                body = std::make_shared<m3t::Body>(name, link_model.mesh, 1.0F, true, false,
                                                   m3t::Transform3fA::Identity());
                body->set_body2world_pose(to_m3t_transform(initial_poses[link_index]));
                const auto body_cache = arm_cache / cache_component(link_model.name);
                add_visual_body(body);
                const auto configured_weight = options.visual_link_weights.find(link_model.name);
                const double visual_weight = configured_weight == options.visual_link_weights.end()
                                                 ? 1.0 : configured_weight->second;
                std::shared_ptr<m3t::RegionModel> region_model;
                std::shared_ptr<m3t::DepthModel> depth_model;
                if (options.use_region || options.use_depth) {
                    std::filesystem::create_directories(body_cache);
                }
                // The geometric models are body-specific; view modalities below
                // share them while retaining their own cameras and renderers.
                if (options.use_region) {
                    region_model = std::make_shared<m3t::RegionModel>(
                        object_name(seed.name, link_model.name, "region_model"), body,
                        body_cache / "region.bin", 0.8F, options.model_divides,
                        options.model_points, 0.05F, 0.002F, false,
                        options.model_image_size);
                }
                if (options.use_depth) {
                    depth_model = std::make_shared<m3t::DepthModel>(
                        object_name(seed.name, link_model.name, "depth_model"), body,
                        body_cache / "depth.bin", 0.8F, options.model_divides,
                        options.model_points, 0.05F, 0.002F, false,
                        options.model_image_size);
                }
                for (const auto& view : views) {
                    if (options.use_region) {
                        const auto modality_name = object_name(
                            seed.name, link_model.name,
                            view_component(view.name, "region_modality"));
                        std::shared_ptr<WeightedVisualModality<m3t::RegionModality>> modality;
                        if (options.region_metadata.empty()) {
                            modality = std::make_shared<WeightedVisualModality<m3t::RegionModality>>(
                                modality_name, body, view.color_camera, region_model);
                            modality->set_n_lines_max(options.model_points);
                        } else {
                            modality = std::make_shared<WeightedVisualModality<m3t::RegionModality>>(
                                modality_name, options.region_metadata, body,
                                view.color_camera, region_model);
                        }
                        modality->set_visual_weight(visual_weight);
                        if (options.measured_occlusions) {
                            modality->MeasureOcclusions(view.depth_camera);
                        } else {
                            modality->DoNotMeasureOcclusions();
                        }
                        if (options.modeled_occlusions) {
                            modality->ModelOcclusions(view.depth_renderer);
                        } else {
                            modality->DoNotModelOcclusions();
                        }
                        if (options.region_checking) {
                            modality->UseRegionChecking(view.region_silhouette_renderer);
                        } else {
                            modality->DoNotUseRegionChecking();
                        }
                        region_modalities.push_back(std::move(modality));
                    }
                    if (options.use_depth) {
                        const auto modality_name = object_name(
                            seed.name, link_model.name,
                            view_component(view.name, "depth_modality"));
                        std::shared_ptr<WeightedVisualModality<m3t::DepthModality>> modality;
                        if (options.depth_metadata.empty()) {
                            modality = std::make_shared<WeightedVisualModality<m3t::DepthModality>>(
                                modality_name, body, view.depth_camera, depth_model);
                            modality->set_n_points_max(options.model_points);
                        } else {
                            modality = std::make_shared<WeightedVisualModality<m3t::DepthModality>>(
                                modality_name, options.depth_metadata, body,
                                view.depth_camera, depth_model);
                        }
                        modality->set_visual_weight(visual_weight);
                        if (options.measured_occlusions) {
                            modality->MeasureOcclusions();
                        } else {
                            modality->DoNotMeasureOcclusions();
                        }
                        if (options.modeled_occlusions) {
                            modality->ModelOcclusions(view.depth_renderer);
                        } else {
                            modality->DoNotModelOcclusions();
                        }
                        if (options.region_checking) {
                            modality->UseSilhouetteChecking(view.body_silhouette_renderer);
                        } else {
                            modality->DoNotUseSilhouetteChecking();
                        }
                        depth_modalities.push_back(std::move(modality));
                    }
                    if (options.use_texture) {
                        const auto modality_name = object_name(
                            seed.name, link_model.name,
                            view_component(view.name, "texture_modality"));
                        std::shared_ptr<WeightedVisualModality<m3t::TextureModality>> modality;
                        if (options.texture_metadata.empty()) {
                            modality = std::make_shared<WeightedVisualModality<m3t::TextureModality>>(
                                modality_name, body, view.color_camera,
                                view.body_silhouette_renderer);
                        } else {
                            modality = std::make_shared<WeightedVisualModality<m3t::TextureModality>>(
                                modality_name, options.texture_metadata, body,
                                view.color_camera, view.body_silhouette_renderer);
                        }
                        modality->set_visual_weight(visual_weight);
                        if (options.measured_occlusions) {
                            modality->MeasureOcclusions(view.depth_camera);
                        } else {
                            modality->DoNotMeasureOcclusions();
                        }
                        if (options.modeled_occlusions) {
                            modality->ModelOcclusions(view.depth_renderer);
                        } else {
                            modality->DoNotModelOcclusions();
                        }
                        texture_modalities.push_back(std::move(modality));
                    }
                }
            }

            const auto identity = m3t::Transform3fA::Identity();
            auto link = std::make_shared<m3t::Link>(
                object_name(seed.name, link_model.name, "link"), body,
                identity, has_encoder_reference ? identity : to_m3t_transform(joint_pose),
                to_m3t_transform(initial_poses[link_index]), free_directions, true);
            for (const auto& modality : region_modalities) {
                if (!link->AddModality(modality)) {
                    failed("could not attach a region modality to a visual link");
                }
            }
            for (const auto& modality : depth_modalities) {
                if (!link->AddModality(modality)) {
                    failed("could not attach a depth modality to a visual link");
                }
            }
            for (const auto& modality : texture_modalities) {
                if (!link->AddModality(modality)) {
                    failed("could not attach a texture modality to a visual link");
                }
            }
            arm.links.emplace(link_model.name, link);
            auto attachment = link;
            if (revolute) {
                JointState joint { *parent_joint_index, link, seed_angle };
                if (has_encoder_reference) {
                    constexpr std::array<bool, 6> fixed {};
                    constexpr std::array<bool, 6> rotation_z { false, false, true, false, false, false };
                    joint.encoder_angle = seed_angle;
                    joint.encoder_reference = std::make_shared<m3t::Link>(
                        object_name(seed.name, link_model.name, "encoder_reference"), nullptr,
                        identity, to_m3t_transform(joint_pose),
                        to_m3t_transform(initial_poses[link_index]), fixed, true);
                    if (!joint.encoder_reference->AddChildLink(link)) {
                        failed("could not attach the physical link to its encoder reference");
                    }
                    joint.encoder_constraint = std::make_shared<m3t::SoftConstraint>(
                        object_name(seed.name, link_model.name, "encoder_prior"),
                        joint.encoder_reference, link, identity, identity, rotation_z, 0.0F, 0.0F);
                    attachment = joint.encoder_reference;
                }
                arm.joints.push_back(std::move(joint));
            }
            for (const auto joint_index : index.child_joint_indices[link_index]) {
                const auto child_index = index.link_indices.at(model.joints[joint_index].child);
                auto child_link = make_link(child_index, joint_index);
                if (!link->AddChildLink(child_link)) {
                    failed("could not attach an M3T child link");
                }
            }
            return attachment;
        };

        arm.root = make_link(index.root_index, std::nullopt);
        if (options.mount_refinement.enabled) {
            const std::array<bool, 6> fixed { false, false, false, false, false, false };
            arm.cad_anchor = std::make_shared<m3t::Link>(
                object_name(seed.name, model.root_link, "cad_anchor"), nullptr,
                m3t::Transform3fA::Identity(), m3t::Transform3fA::Identity(),
                to_m3t_transform(initial_poses[index.root_index]), fixed, true);
            if (!arm.cad_anchor->AddChildLink(arm.root)) {
                failed("could not attach the physical base to its CAD anchor");
            }
        }
        const auto optimizer_root = arm.cad_anchor ? arm.cad_anchor : arm.root;
        const auto optimizer_name =
            object_name(seed.name, model.root_link, "optimizer");
        if (options.optimizer_metadata.empty()) {
            arm.optimizer = std::make_shared<m3t::Optimizer>(optimizer_name, optimizer_root);
        } else {
            arm.optimizer = std::make_shared<m3t::Optimizer>(
                optimizer_name, options.optimizer_metadata, optimizer_root);
        }
        if (options.encoder_reference_links) {
            for (const auto& joint : arm.joints) {
                if (!arm.optimizer->AddSoftConstraint(joint.encoder_constraint)) {
                    failed("could not attach an encoder reference soft constraint");
                }
            }
        }
        if (options.mount_refinement.enabled) {
            const std::array<bool, 6> all { true, true, true, true, true, true };
            const auto constraint = std::make_shared<m3t::SoftConstraint>(
                object_name(seed.name, model.root_link, "cad_mount_prior"),
                arm.cad_anchor, arm.root, m3t::Transform3fA::Identity(),
                m3t::Transform3fA::Identity(), all, 0.0F, 0.0F,
                static_cast<float>(options.mount_refinement.rotation_standard_deviation_rad),
                static_cast<float>(options.mount_refinement.translation_standard_deviation_m));
            if (!arm.optimizer->AddSoftConstraint(constraint)) {
                failed("could not attach the CAD mount soft constraint");
            }
        }
        for (const auto& constraint_options : options.constraints) {
            if (constraint_options.arm != seed.name) {
                continue;
            }
            if (constraint_options.soft) {
                const auto constraint = std::make_shared<m3t::SoftConstraint>(
                    constraint_options.name, constraint_options.metadata,
                    arm.links.at(constraint_options.link1),
                    arm.links.at(constraint_options.link2));
                if (!arm.optimizer->AddSoftConstraint(constraint)) {
                    failed("could not attach M3T soft constraint to its arm optimizer");
                }
            } else {
                const auto constraint = std::make_shared<m3t::Constraint>(
                    constraint_options.name, constraint_options.metadata,
                    arm.links.at(constraint_options.link1),
                    arm.links.at(constraint_options.link2));
                if (!arm.optimizer->AddConstraint(constraint)) {
                    failed("could not attach M3T constraint to its arm optimizer");
                }
            }
        }
        if (!tracker->AddOptimizer(arm.optimizer)) {
            failed("could not add an M3T arm optimizer");
        }
        if (!options.encoder_reference_links) {
            arm.static_detector = std::make_shared<m3t::StaticDetector>(
                object_name(seed.name, model.root_link, "static_detector"),
                arm.optimizer, to_m3t_transform(initial_poses[index.root_index]), false);
            if (!tracker->AddDetector(arm.static_detector)) {
                failed("could not add an M3T seed detector");
            }
        }
        arms.push_back(std::move(arm));
    }

    void add_refiner() {
        if (!options.refine_initialization) {
            return;
        }
        if (options.refiner_metadata.empty()) {
            refiner = std::make_shared<m3t::Refiner>(
                "rbt_m3t_initial_refiner", options.correspondence_iterations,
                options.update_iterations, 1);
        } else {
            refiner = std::make_shared<m3t::Refiner>(
                "rbt_m3t_initial_refiner", options.refiner_metadata);
        }
        for (const auto& arm : arms) {
            if (!refiner->AddOptimizer(arm.optimizer)) {
                failed("could not add an arm optimizer to the M3T refiner");
            }
        }
        if (!tracker->AddRefiner(refiner)) {
            failed("could not add the M3T initialization refiner");
        }
    }

    [[nodiscard]] std::vector<ValidatedJointPrior> validate_tracking_priors(
        const std::vector<ArmSeed>& active_seeds,
        const std::vector<JointAnglePrior>& priors) const {
        auto validated = validate_joint_priors(model, index, active_seeds, priors);
        if (validated.size() != active_seeds.size() * index.revolute_joint_indices.size()) {
            if (options.encoder_reference_links) {
                invalid("encoder reference links require fresh feedback for every arm's revolute joints");
            }
            if (options.mount_refinement.enabled) {
                invalid("mount refinement requires fresh feedback for every arm's revolute joints");
            }
        }
        if (options.encoder_reference_links) {
            for (const auto& prior : validated) {
                // Upstream SoftConstraint forms the variance and reciprocal in float.
                const float variance = prior.standard_deviation * prior.standard_deviation;
                const float precision = 1.0F / variance;
                if (!std::isfinite(precision) || precision <= 0.0F) {
                    invalid("encoder reference prior precision must be positive and float-representable");
                }
            }
        }
        return validated;
    }

    JointState& joint_state(ArmState& arm, std::size_t joint_index) {
        const auto joint = std::find_if(arm.joints.begin(), arm.joints.end(),
            [joint_index](const JointState& candidate) {
                return candidate.joint_index == joint_index;
            });
        if (joint == arm.joints.end()) {
            failed("validated joint prior has no M3T articulated link");
        }
        return *joint;
    }

    void set_encoder_correction(JointState& joint, double correction) {
        const auto& model_joint = model.joints[joint.joint_index];
        joint.angle = std::clamp(joint.encoder_angle + correction,
                                 model_joint.lower, model_joint.upper);
        joint.link->set_joint2parent_pose(to_m3t_transform(
            revolute_rotation(joint.angle - joint.encoder_angle)));
    }

    void project_joints() {
        if (options.mount_refinement.enabled) {
            // Upstream composes float SE(3) increments. Project their rotation
            // back onto SO(3) before propagating, so a stationary mount does not
            // accumulate scale/shear over thousands of optimization steps.
            for (auto& arm : arms) {
                auto correction = arm.root->joint2parent_pose();
                Eigen::Quaternionf rotation(correction.linear());
                if (!rotation.coeffs().array().isFinite().all()
                    || rotation.norm() < 1.0e-6F) {
                    failed("M3T produced an invalid mount correction rotation");
                }
                rotation.normalize();
                correction.linear() = rotation.toRotationMatrix();
                arm.root->set_joint2parent_pose(correction);
            }
            if (locks_joints_to_feedback(options)) {
                return; // Preserve exact encoder doubles; no optimized joint coordinates exist.
            }
        }
        for (auto& arm : arms) {
            for (auto& joint : arm.joints) {
                const auto& model_joint = model.joints[joint.joint_index];
                const auto local_pose = to_eigen_affine(joint.link->joint2parent_pose());
                if (joint.encoder_reference) {
                    const double correction = detail::unwrap_and_clamp_joint_angle(
                        z_rotation_angle(local_pose), joint.angle - joint.encoder_angle,
                        -kMaximumEncoderCorrection, kMaximumEncoderCorrection);
                    set_encoder_correction(joint, correction);
                } else {
                    joint.angle = detail::unwrap_and_clamp_joint_angle(
                        z_rotation_angle(model_joint.origin.inverse() * local_pose), joint.angle,
                        model_joint.lower, model_joint.upper);
                    joint.link->set_joint2parent_pose(to_m3t_transform(
                        joint_transform(model_joint, joint.angle)));
                }
            }
        }
    }

    void apply_encoder_reference_feedback(const std::vector<ValidatedJointPrior>& priors) {
        for (const auto& prior : priors) {
            auto& joint = joint_state(arms[prior.arm_index], prior.joint_index);
            const auto& model_joint = model.joints[joint.joint_index];
            const double correction = joint.angle - joint.encoder_angle;
            joint.encoder_angle = prior.measured_angle;
            joint.encoder_reference->set_joint2parent_pose(to_m3t_transform(
                joint_transform(model_joint, joint.encoder_angle)));
            joint.encoder_constraint->set_standard_deviation_rotation(prior.standard_deviation);
            set_encoder_correction(joint, correction);
        }
        if (!tracker->CalculateConsistentPoses()) {
            failed("M3T could not propagate encoder reference joint poses");
        }
    }

    void apply_locked_joint_feedback(const std::vector<ValidatedJointPrior>& priors) {
        for (const auto& prior : priors) {
            auto& joint = joint_state(arms[prior.arm_index], prior.joint_index);
            joint.angle = prior.measured_angle;
            joint.link->set_joint2parent_pose(to_m3t_transform(
                joint_transform(model.joints[joint.joint_index], joint.angle)));
        }
        if (!tracker->CalculateConsistentPoses()) {
            failed("M3T could not propagate locked encoder joint poses");
        }
    }

    void clear_joint_angle_priors() {
        if (options.encoder_reference_links) {
            return;
        }
        for (auto& arm : arms) {
            arm.optimizer->ClearJointAnglePriors();
        }
    }

    void install_joint_angle_priors(
        const std::vector<ValidatedJointPrior>& validated_priors) {
        if (options.mount_refinement.enabled || options.encoder_reference_links) {
            return;
        }
        for (std::size_t arm_index = 0; arm_index < arms.size(); ++arm_index) {
            auto& arm = arms[arm_index];
            std::vector<m3t::JointAnglePrior> optimizer_priors;
            for (const auto& prior : validated_priors) {
                if (prior.arm_index != arm_index) {
                    continue;
                }
                const auto& joint = joint_state(arm, prior.joint_index);
                optimizer_priors.push_back(m3t::JointAnglePrior {
                    joint.link->first_jacobian_index(),
                    static_cast<float>(joint.angle),
                    prior.target_angle,
                    prior.standard_deviation,
                });
            }
            if (!arm.optimizer->SetJointAnglePriors(std::move(optimizer_priors))) {
                clear_joint_angle_priors();
                failed("M3T could not apply a validated joint angle prior");
            }
        }
    }

    bool execute_detecting_step(
        int iteration, const std::vector<ValidatedJointPrior>& validated_priors) {
        // Destroy callbacks before their captured detection-local targets expire,
        // and discard residuals on success, failure, or exception.
        struct DetectionCleanup {
            Impl& engine;
            ~DetectionCleanup() {
                if (engine.refiner) {
                    engine.refiner->set_before_optimization_callback({});
                    engine.refiner->set_after_optimization_callback({});
                }
                engine.clear_joint_angle_priors();
            }
        } cleanup { *this };
        if (refiner && !validated_priors.empty()) {
            refiner->set_before_optimization_callback([this, &validated_priors] {
                project_joints();
                install_joint_angle_priors(validated_priors);
                return true;
            });
            refiner->set_after_optimization_callback([this] {
                project_joints();
                return tracker->CalculateConsistentPoses();
            });
        }
        tracker->ExecuteDetection(true);
        return tracker->ExecuteDetectingStep(iteration);
    }

    TrackingEstimate estimates() const {
        TrackingEstimate result;
        result.joints_locked_to_feedback = locks_joints_to_feedback(options);
        result.encoder_reference_links = options.encoder_reference_links;
        result.visual_link_weights = options.visual_link_weights;
        result.modalities.clear();
        if (options.use_region) {
            result.modalities.emplace_back("region");
        }
        if (options.use_depth) {
            result.modalities.emplace_back("depth");
        }
        if (options.use_texture) {
            result.modalities.emplace_back("texture");
        }
        const Eigen::Affine3f world_to_camera =
            views.front().images->camera_in_world.inverse();
        for (const auto& arm : arms) {
            if (!is_rigid_pose(arm.root->link2world_pose())) {
                failed("M3T produced a non-finite or non-rigid arm base pose");
            }
            for (const auto& model_link : model.links) {
                const auto link = arm.links.at(model_link.name);
                const auto pose = world_to_camera * to_eigen_affine(link->link2world_pose());
                if (!is_rigid_pose(pose)) {
                    failed("M3T produced a non-finite or non-rigid link pose");
                }
                result.links.push_back(LinkEstimate { arm.seed.name, model_link.name, pose });
            }
            if (options.mount_refinement.enabled) {
                const Eigen::Affine3f refined =
                    world_to_camera * to_eigen_affine(arm.root->link2world_pose());
                const Eigen::Affine3f correction = arm.seed.base_in_camera.inverse() * refined;
                result.mount_refinement.push_back(MountRefinementEstimate {
                    arm.seed.name, arm.seed.base_in_camera, refined,
                    static_cast<double>(correction.translation().norm()),
                    static_cast<double>(Eigen::AngleAxisf(correction.rotation()).angle()),
                });
            }
            auto& angles = result.joint_angles[arm.seed.name];
            for (const auto& joint : arm.joints) {
                angles.emplace(model.joints[joint.joint_index].name, joint.angle);
            }
        }
        return result;
    }

    RobotModel model;
    std::vector<ArmSeed> seeds;
    EngineOptions options;
    std::vector<ViewState> views;
    ModelIndex index;
    std::shared_ptr<m3t::Tracker> tracker;
    std::shared_ptr<m3t::RendererGeometry> renderer_geometry;
    std::shared_ptr<m3t::Refiner> refiner;
    std::vector<std::shared_ptr<m3t::Body>> target_bodies;
    std::vector<ArmState> arms;
    std::thread::id owner_thread;
    std::size_t next_visual_body_id { 1U };
    bool set_up {};
    bool started {};
};

TrackerEngine::TrackerEngine(RobotModel model, std::vector<ArmSeed> seeds,
                             EngineOptions options,
                             std::shared_ptr<CameraImages> images,
                             std::shared_ptr<CameraImages> secondary_images)
    : impl_{std::make_unique<Impl>(std::move(model), std::move(seeds),
                                   std::move(options), std::move(images),
                                   std::move(secondary_images))} {}

TrackerEngine::~TrackerEngine() = default;

void TrackerEngine::set_up() {
    impl_->ensure_owner_thread();
    if (impl_->set_up) {
        return;
    }
    impl_->validate_views("setup");
    impl_->prepare_renderers();
    for (std::size_t i = 0; i < impl_->seeds.size(); ++i) {
        impl_->build_arm(impl_->seeds[i], i);
    }
    impl_->add_refiner();
    if (!impl_->tracker->SetUp()) {
        failed("could not set up the articulated M3T tracker");
    }
    impl_->set_up = true;
}

void TrackerEngine::reinitialize(
    const std::vector<ArmSeed>& seeds, int iteration,
    const std::vector<JointAnglePrior>& priors) {
    impl_->ensure_owner_thread();
    if (!impl_->set_up) {
        throw std::logic_error("set up the M3T engine before reinitializing it");
    }
    if (iteration < 0) {
        invalid("M3T iteration index must be non-negative");
    }
    impl_->validate_views("reinitialization");
    if (seeds.size() != impl_->arms.size()) {
        invalid("M3T reinitialization must provide exactly the constructed arms");
    }

    std::map<std::string, ArmSeed> requested_seeds;
    for (const auto& seed : seeds) {
        const auto arm = std::find_if(impl_->arms.begin(), impl_->arms.end(),
                                      [&seed](const Impl::ArmState& candidate) {
                                          return candidate.seed.name == seed.name;
                                      });
        if (arm == impl_->arms.end()) {
            invalid("M3T reinitialization arm identities must match the constructed arms");
        }
        if (seed.fixed_base != arm->seed.fixed_base) {
            invalid("M3T reinitialization may not change an arm's fixed-base choice");
        }
        (void)validate_seed(impl_->model, impl_->index, seed);
        if (!requested_seeds.emplace(seed.name, seed).second) {
            invalid("M3T reinitialization arm names must be unique");
        }
    }
    // Prior indices follow the constructed arm order, even when callers reorder seeds.
    std::vector<ArmSeed> ordered_seeds;
    ordered_seeds.reserve(impl_->arms.size());
    for (const auto& arm : impl_->arms) {
        ordered_seeds.push_back(requested_seeds.at(arm.seed.name));
    }
    const auto validated_priors = impl_->validate_tracking_priors(ordered_seeds, priors);
    impl_->clear_joint_angle_priors();
    if (!impl_->tracker->UpdateCameras(iteration)) {
        failed("M3T cameras could not read the caller-selected RGB-D views for reinitialization");
    }

    for (auto& arm : impl_->arms) {
        auto seed = requested_seeds.at(arm.seed.name);
        arm.seed = seed;
        const Eigen::Affine3f base_in_world =
            impl_->views.front().images->camera_in_world * seed.base_in_camera;
        if (impl_->options.mount_refinement.enabled) {
            arm.cad_anchor->set_link2world_pose(to_m3t_transform(base_in_world));
            arm.root->set_joint2parent_pose(m3t::Transform3fA::Identity());
        } else {
            arm.root->set_link2world_pose(to_m3t_transform(base_in_world));
            if (arm.root->body_ptr()) {
                arm.root->body_ptr()->set_body2world_pose(to_m3t_transform(base_in_world));
            }
        }
        if (arm.static_detector) {
            arm.static_detector->set_link2world_pose(to_m3t_transform(base_in_world));
        }
        for (auto& joint : arm.joints) {
            const auto& model_joint = impl_->model.joints[joint.joint_index];
            joint.angle = seed.joint_angles.at(model_joint.name);
            if (joint.encoder_reference) {
                joint.encoder_angle = joint.angle;
                joint.link->set_joint2parent_pose(m3t::Transform3fA::Identity());
            } else {
                joint.link->set_joint2parent_pose(to_m3t_transform(
                    joint_transform(model_joint, joint.angle)));
            }
        }
    }
    impl_->seeds = seeds;
    if (impl_->options.encoder_reference_links) {
        impl_->apply_encoder_reference_feedback(validated_priors);
        impl_->tracker->StartTracking();
    } else {
        if (impl_->options.mount_refinement.enabled) {
            impl_->apply_locked_joint_feedback(validated_priors);
        }
        if (!impl_->execute_detecting_step(iteration, validated_priors)) {
            failed("M3T seeded detection or initialization refinement failed");
        }
        impl_->project_joints();
        if (!impl_->tracker->CalculateConsistentPoses()) {
            failed("M3T could not synchronize reinitialized articulated poses");
        }
    }
    if (!impl_->tracker->ExecuteStartingStep(iteration)) {
        failed("M3T could not restart tracking after reinitialization");
    }
    impl_->started = true;
}

TrackingEstimate TrackerEngine::step(
    int iteration, const std::vector<JointAnglePrior>& priors) {
    impl_->ensure_owner_thread();
    if (!impl_->set_up) {
        throw std::logic_error("set up the M3T engine before stepping it");
    }
    if (iteration < 0) {
        invalid("M3T iteration index must be non-negative");
    }
    impl_->validate_views("step");
    std::vector<ArmSeed> active_seeds;
    active_seeds.reserve(impl_->arms.size());
    for (const auto& arm : impl_->arms) {
        active_seeds.push_back(arm.seed);
    }
    const auto validated_priors = impl_->validate_tracking_priors(active_seeds, priors);
    impl_->clear_joint_angle_priors();
    if (!impl_->tracker->UpdateCameras(iteration)) {
        failed("M3T cameras could not read the caller-selected RGB-D views");
    }
    if (impl_->options.encoder_reference_links) {
        impl_->apply_encoder_reference_feedback(validated_priors);
    } else if (impl_->options.mount_refinement.enabled) {
        impl_->apply_locked_joint_feedback(validated_priors);
    }
    if (!impl_->started) {
        if (impl_->options.use_static_detector || impl_->options.refine_initialization) {
            if (!impl_->execute_detecting_step(iteration, validated_priors)) {
                failed("M3T seeded detection or initialization refinement failed");
            }
            impl_->project_joints();
            if (!impl_->tracker->CalculateConsistentPoses()) {
                failed("M3T could not synchronize initial articulated poses");
            }
        } else {
            if (!impl_->tracker->CalculateConsistentPoses()) {
                failed("M3T could not synchronize initial articulated poses");
            }
            impl_->tracker->StartTracking();
        }
        if (!impl_->tracker->ExecuteStartingStep(iteration)) {
            failed("M3T could not start tracking modalities");
        }
        impl_->started = true;
    }

    for (int corr_iteration = 0;
         corr_iteration < impl_->options.correspondence_iterations;
         ++corr_iteration) {
        if (!impl_->tracker->CalculateCorrespondences(iteration, corr_iteration)) {
            failed("M3T could not calculate correspondences");
        }
        if (impl_->options.visualize_correspondences) {
            const auto save_index = static_cast<std::uint64_t>(iteration)
                                    * static_cast<std::uint64_t>(
                                        impl_->options.correspondence_iterations)
                                    + static_cast<std::uint64_t>(corr_iteration);
            if (!impl_->tracker->VisualizeCorrespondences(
                    checked_visualization_save_index(save_index))) {
                failed("M3T correspondence visualization failed");
            }
        }
        for (int update_iteration = 0;
             update_iteration < impl_->options.update_iterations;
             ++update_iteration) {
            impl_->install_joint_angle_priors(validated_priors);
            if (!impl_->tracker->CalculateGradientAndHessian(
                    iteration, corr_iteration, update_iteration)
                || !impl_->tracker->CalculateOptimization(
                    iteration, corr_iteration, update_iteration)) {
                failed("M3T articulated optimization failed");
            }
            impl_->project_joints();
            if (!impl_->tracker->CalculateConsistentPoses()) {
                failed("M3T could not restore exact articulated poses");
            }
            if (impl_->options.visualize_optimization) {
                const auto save_index =
                    (static_cast<std::uint64_t>(iteration)
                         * static_cast<std::uint64_t>(
                             impl_->options.correspondence_iterations)
                     + static_cast<std::uint64_t>(corr_iteration))
                        * static_cast<std::uint64_t>(
                            impl_->options.update_iterations)
                    + static_cast<std::uint64_t>(update_iteration);
                if (!impl_->tracker->VisualizeOptimization(
                        checked_visualization_save_index(save_index))) {
                    failed("M3T optimization visualization failed");
                }
            }
        }
    }
    if (!impl_->tracker->CalculateResults(iteration)) {
        failed("M3T could not calculate tracking results");
    }
    if (impl_->options.visualize_results
        && !impl_->tracker->VisualizeResults(iteration)) {
        failed("M3T result visualization failed");
    }
    auto result = impl_->estimates();
    result.joint_prior.enabled = !priors.empty();
    result.joint_prior.status = priors.empty() ? "disabled" : "applied";
    result.joint_prior.priors = priors;
    return result;
}

} // namespace rbt_m3t
