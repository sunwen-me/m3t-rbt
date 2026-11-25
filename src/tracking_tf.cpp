#include <rbt_m3t/tracking_tf.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rbt_m3t {
namespace {

[[noreturn]] void invalid(const std::string& reason) {
    throw std::invalid_argument("tracking TF: " + reason);
}

rbt::tf::FrameId frame_id(std::string_view name) {
    try {
        return rbt::tf::FrameId(name);
    } catch (const rbt::tf::TfError&) {
        invalid("invalid TF frame name: " + std::string(name));
    }
}

void validate_rigid(const Eigen::Affine3f& pose) {
    // M3T poses are native floats. Validate before normalizing the quaternion;
    // normalization must only absorb rounding, never repair scale or reflection.
    constexpr double tolerance = 1.0e-4;
    const Eigen::Matrix4d matrix = pose.matrix().cast<double>();
    const Eigen::Matrix3d rotation = matrix.topLeftCorner<3, 3>();
    if (!matrix.allFinite()
        || (matrix.row(3) - Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0))
                   .cwiseAbs().maxCoeff() > tolerance
        || (rotation.transpose() * rotation - Eigen::Matrix3d::Identity())
                   .cwiseAbs().maxCoeff() > tolerance
        || std::abs(rotation.determinant() - 1.0) > tolerance) {
        invalid("pose must be a finite proper rigid transform");
    }
}

rbt::tf::Transform3 native_transform(const Eigen::Affine3d& pose) {
    Eigen::Quaterniond quaternion(pose.linear());
    quaternion.normalize();
    return rbt::tf::Transform3(
        rbt::tf::Vec3(pose.translation().x(), pose.translation().y(), pose.translation().z()),
        rbt::tf::Quaternion(quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w()));
}

// Returns parent-before-child joints, independently of the input joint order.
std::vector<const ModelJoint*> validate_model(const RobotModel& model) {
    std::set<std::string> links;
    for (const auto& link : model.links) {
        (void)frame_id(link.name);
        if (!links.insert(link.name).second) {
            invalid("duplicate model link: " + link.name);
        }
    }
    (void)frame_id(model.root_link);
    if (!links.contains(model.root_link)) {
        invalid("model root is missing");
    }
    std::set<std::string> joint_names, children;
    std::map<std::string, std::vector<const ModelJoint*>> outgoing;
    for (const auto& joint : model.joints) {
        (void)frame_id(joint.name);
        if (!joint_names.insert(joint.name).second) {
            invalid("duplicate model joint: " + joint.name);
        }
        if (!links.contains(joint.parent) || !links.contains(joint.child)) {
            invalid("joint references an unknown model link");
        }
        if (joint.parent == joint.child || joint.child == model.root_link
            || !children.insert(joint.child).second) {
            invalid("model must be a tree with exactly one parent per non-root link");
        }
        validate_rigid(joint.origin);
        outgoing[joint.parent].push_back(&joint);
    }
    if (model.joints.size() != links.size() - 1U) {
        invalid("model tree is incomplete");
    }
    std::vector<const ModelJoint*> ordered;
    std::vector<std::string> pending { model.root_link };
    std::set<std::string> visited { model.root_link };
    for (std::size_t index = 0; index < pending.size(); ++index) {
        const auto found = outgoing.find(pending[index]);
        if (found == outgoing.end()) {
            continue;
        }
        for (const auto* joint : found->second) {
            if (!visited.insert(joint->child).second) {
                invalid("model graph contains a cycle");
            }
            ordered.push_back(joint);
            pending.push_back(joint->child);
        }
    }
    if (visited.size() != links.size()) {
        invalid("model graph is disconnected or cyclic");
    }
    return ordered;
}

} // namespace

std::vector<rbt::tf::DynamicRecord> tracking_tf_records(
    const RobotModel& model,
    const TrackingEstimate& estimate,
    const RectifiedPair& pair,
    const rbt::time::Timestamp& now,
    std::string_view frame_prefix,
    std::span<const ArmSeed> nominal_arms) {
    if (!pair.timing.capture) {
        return {};
    }
    const auto& capture = *pair.timing.capture;
    if (!capture.clock.checked_compatible(now.clock).has_value() || capture.ns > now.ns) {
        return {};
    }
    // Ordered signed timestamps may span INT64_MIN..INT64_MAX. Unsigned
    // subtraction preserves their nonnegative difference without signed overflow.
    const auto age = static_cast<std::uint64_t>(now.ns)
                     - static_cast<std::uint64_t>(capture.ns);
    if (age > 500'000'000U) {
        return {};
    }

    const auto joints = validate_model(model);
    const auto camera_frame = frame_id(pair.frame_id);
    if (!frame_prefix.empty()) {
        (void)frame_id(frame_prefix);
    }
    if (estimate.encoder_reference_links) {
        if (estimate.joints_locked_to_feedback) {
            invalid("encoder-reference links require visually optimized joints");
        }
    } else if (!estimate.joints_locked_to_feedback) {
        invalid("estimate joints must be locked to feedback");
    }
    if (estimate.joint_angles.empty() || estimate.joint_angles.size() > 2U) {
        invalid("estimate must contain one or two arms");
    }

    using LinkPoses = std::map<std::string, const Eigen::Affine3f*>;
    std::map<std::string, LinkPoses> arms;
    for (const auto& [arm, angles] : estimate.joint_angles) {
        (void)angles; // Joint angles do not replace the observed link poses.
        (void)frame_id(arm);
        arms.emplace(arm, LinkPoses {});
    }
    std::set<std::string> model_links;
    for (const auto& link : model.links) {
        model_links.insert(link.name);
    }
    for (const auto& link : estimate.links) {
        const auto arm = arms.find(link.arm);
        if (arm == arms.end() || !model_links.contains(link.link)) {
            invalid("estimate references an unknown arm or model link");
        }
        if (!arm->second.emplace(link.link, &link.in_camera).second) {
            invalid("duplicate estimated arm/link");
        }
        validate_rigid(link.in_camera);
    }

    for (const auto& [arm, poses] : arms) {
        if (poses.size() != model.links.size()) {
            invalid("each arm must estimate every model link");
        }
    }

    std::map<std::string, const Eigen::Affine3f*> nominal_poses;
    if (estimate.encoder_reference_links) {
        for (const auto& nominal : nominal_arms) {
            if (!arms.contains(nominal.name)
                || !nominal_poses.emplace(nominal.name, &nominal.base_in_camera).second) {
                invalid("unknown or duplicate nominal CAD arm");
            }
            validate_rigid(nominal.base_in_camera);
        }
        if (nominal_poses.size() != arms.size()) {
            invalid("each encoder-reference arm requires exactly one nominal CAD pose");
        }
    }
    if (!estimate.mount_refinement.empty()) {
        std::set<std::string> mount_arms;
        for (const auto& mount : estimate.mount_refinement) {
            const auto arm = arms.find(mount.arm);
            if (arm == arms.end() || !mount_arms.insert(mount.arm).second) {
                invalid("unknown or duplicate mount refinement arm");
            }
            // A separately configured TF CAD anchor can differ from the mount
            // optimizer's seed/prior center. Keep the former as the TF parent.
            if (!estimate.encoder_reference_links) {
                nominal_poses.emplace(mount.arm, &mount.nominal_base_in_camera);
            }
            validate_rigid(mount.nominal_base_in_camera);
            validate_rigid(mount.refined_base_in_camera);
            if (!std::isfinite(mount.translation_correction_m) || mount.translation_correction_m < 0.0
                || !std::isfinite(mount.rotation_correction_rad) || mount.rotation_correction_rad < 0.0) {
                invalid("mount correction diagnostics must be finite and nonnegative");
            }
            // Use an absolute tolerance so large camera offsets cannot mask a different root.
            constexpr double root_tolerance = 1.0e-4;
            const auto& root = *arm->second.at(model.root_link);
            if ((mount.refined_base_in_camera.matrix().cast<double>()
                 - root.matrix().cast<double>()).cwiseAbs().maxCoeff() > root_tolerance) {
                invalid("refined mount diagnostic disagrees with estimated root");
            }
            const Eigen::Affine3d nominal = mount.nominal_base_in_camera.cast<double>();
            const Eigen::Affine3d refined = mount.refined_base_in_camera.cast<double>();
            const double translation = (refined.translation() - nominal.translation()).norm();
            const Eigen::Matrix3d relative_rotation = refined.linear() * nominal.linear().transpose();
            const double sine = 0.5 * Eigen::Vector3d {
                relative_rotation(2, 1) - relative_rotation(1, 2),
                relative_rotation(0, 2) - relative_rotation(2, 0),
                relative_rotation(1, 0) - relative_rotation(0, 1) }.norm();
            const double rotation = std::atan2(sine,
                std::clamp((relative_rotation.trace() - 1.0) / 2.0, -1.0, 1.0));
            if (std::abs(mount.translation_correction_m - translation) > root_tolerance
                || std::abs(mount.rotation_correction_rad - rotation) > root_tolerance) {
                invalid("mount correction diagnostics disagree with nominal and refined poses");
            }
        }
        if (mount_arms.size() != arms.size()) {
            invalid("each arm requires exactly one mount refinement");
        }
    } else if (!estimate.encoder_reference_links) {
        invalid("each arm requires exactly one mount refinement");
    }

    // Validate every generated ID before constructing any output, including
    // collisions caused by hierarchical names across different arms.
    std::map<std::string, std::map<std::string, rbt::tf::FrameId>> frames;
    std::map<std::string, rbt::tf::FrameId> mount_frames;
    std::set<std::string> generated;
    const auto prefix = frame_prefix.empty() ? std::string {} : std::string(frame_prefix) + '/';
    const auto generated_frame = [&](const std::string& name) {
        auto id = frame_id(name); // FrameId enforces the 64-byte wire limit.
        if (id == camera_frame || !generated.insert(name).second) {
            invalid("generated TF frame collides with camera or another generated frame");
        }
        return id;
    };
    for (const auto& [arm, poses] : arms) {
        (void)poses;
        mount_frames.emplace(arm, generated_frame(prefix + arm + "/cad_mount"));
        auto& arm_frames = frames[arm];
        for (const auto& link : model.links) {
            arm_frames.emplace(link.name, generated_frame(prefix + arm + '/' + link.name));
        }
    }

    std::vector<rbt::tf::DynamicRecord> records;
    records.reserve(arms.size() * (model.links.size() + 1U));
    for (const auto& [arm, poses] : arms) {
        const auto& arm_frames = frames.at(arm);
        const Eigen::Affine3d nominal = nominal_poses.at(arm)->cast<double>();
        const Eigen::Affine3d root = poses.at(model.root_link)->cast<double>();
        // Intentionally sample the nominal anchor too: every edge has exactly
        // the same capture lifetime, uncertainty and activation epoch.
        records.push_back({ camera_frame, mount_frames.at(arm), capture,
                            pair.timing.capture_uncertainty_ns, 0U, native_transform(nominal) });
        records.push_back({ mount_frames.at(arm), arm_frames.at(model.root_link), capture,
                            pair.timing.capture_uncertainty_ns, 0U,
                            native_transform(nominal.inverse() * root) });
        for (const auto* joint : joints) {
            const Eigen::Affine3d parent = poses.at(joint->parent)->cast<double>();
            const Eigen::Affine3d child = poses.at(joint->child)->cast<double>();
            const Eigen::Affine3d relative = parent.inverse() * child;
            records.push_back({ arm_frames.at(joint->parent), arm_frames.at(joint->child),
                                capture, pair.timing.capture_uncertainty_ns, 0U,
                                native_transform(relative) });
        }
    }
    return records;
}

TrackingTfPublishResult publish_tracking_tf_batch(
    rbt::tf::ScopedDynamicTfPublisher<TrackingTfCapability>& publisher,
    std::span<const rbt::tf::DynamicRecord> records,
    rbt::time::HostClock& clock,
    std::stop_token stop) {
    using namespace std::chrono_literals;
    TrackingTfPublishResult result;
    const auto deadline = std::chrono::steady_clock::now() + 25ms;
    for (const auto& record : records) {
        for (;;) {
            if (stop.stop_requested()) return result;
            const auto now = clock.now_monotonic();
            if (!now) throw std::runtime_error("tracking TF: publish clock unavailable");
            const auto age = rbt::time::checked_difference(now.value(), record.stamp);
            if (!age || age.value() < 0 || age.value() > 500'000'000) return result;
            try {
                publisher.publish(record);
                ++result.published_records;
                break;
            } catch (const rbt::tf::TfTransportError& error) {
                if (error.stream_error() != rbt::StreamError::QueueFull) throw;
                ++result.queue_full_events;
                if (std::chrono::steady_clock::now() >= deadline) return result;
                std::this_thread::sleep_for(500us);
            }
        }
    }
    return result;
}

} // namespace rbt_m3t
