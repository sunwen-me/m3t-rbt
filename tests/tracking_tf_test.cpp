#include <rbt_m3t/tracking_tf.hpp>

#include <rbt/tf_wire.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

rbt::time::ClockIdentity clock_id(std::uint8_t source = 1U,
                                 std::uint64_t generation = 3U,
                                 rbt::time::ClockKind kind = rbt::time::ClockKind::HostMonotonic) {
    std::array<std::uint8_t, 48> bytes {};
    bytes[0] = source;
    auto result = rbt::time::ClockIdentity::create(
        kind, rbt::time::ClockSourceId(bytes), generation);
    check(result.has_value(), "valid fixture clock");
    return std::move(result).value();
}

Eigen::Affine3f pose(float x, float y, float z, float rx, float ry, float rz) {
    Eigen::Affine3f result = Eigen::Affine3f::Identity();
    result.translation() = Eigen::Vector3f(x, y, z);
    result.linear() = (Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX())
                       * Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY())
                       * Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ())).toRotationMatrix();
    return result;
}

Eigen::Affine3d eigen_pose(const rbt::tf::Transform3& transform) {
    const auto t = transform.translation();
    const auto q = transform.rotation();
    Eigen::Affine3d result = Eigen::Affine3d::Identity();
    result.translation() = Eigen::Vector3d(t.x(), t.y(), t.z());
    result.linear() = Eigen::Quaterniond(q.w(), q.x(), q.y(), q.z()).toRotationMatrix();
    return result;
}

void check_pose(const rbt::tf::Transform3& actual, const Eigen::Affine3d& expected,
                const std::string& message, double tolerance = 3.0e-6) {
    check((eigen_pose(actual).matrix() - expected.matrix()).cwiseAbs().maxCoeff() < tolerance,
          message);
}

struct Fixture {
    rbt_m3t::RobotModel model;
    rbt_m3t::TrackingEstimate estimate;
    std::vector<rbt_m3t::ArmSeed> nominal_arms;
    rbt_m3t::RectifiedPair pair {};
    rbt::time::Timestamp now { clock_id(), 1'100'000'000 };

    Fixture() {
        model.name = "TF CPU fixture";
        model.root_link = "base_link";
        model.links = { { "base_link", {} }, { "elbow", {} },
                        { "wrist", {} }, { "gripper", {} } };
        // Deliberately not topologically sorted.
        model.joints = { joint("grip", "wrist", "gripper"),
                         joint("shoulder", "base_link", "elbow"),
                         joint("forearm", "elbow", "wrist") };
        for (const std::string arm : { "so101_left", "so101_right" }) {
            const float sign = arm == "so101_left" ? 1.0F : -1.0F;
            const auto base = pose(sign * 0.31F, -0.24F, 0.72F, 0.3F, -0.4F, sign * 0.6F);
            const Eigen::Affine3f elbow = base * pose(0.1F, 0.2F, -0.03F, 0.7F, 0.1F, -0.2F);
            const Eigen::Affine3f wrist = elbow * pose(0.04F, -0.12F, 0.2F, -0.3F, 0.5F, 0.4F);
            const Eigen::Affine3f gripper = wrist * pose(0.09F, 0.02F, 0.08F, 0.2F, -0.6F, 0.1F);
            // Also scramble estimates: neither output nor validation may depend on order.
            const auto nominal = pose(sign * 0.29F, -0.22F, 0.70F, 0.2F, -0.3F, sign * 0.5F);
            const Eigen::Affine3f correction = nominal.inverse() * base;
            estimate.mount_refinement.push_back({ arm, nominal, base,
                static_cast<double>(correction.translation().norm()),
                static_cast<double>(Eigen::AngleAxisf(correction.rotation()).angle()) });
            estimate.links.push_back({ arm, "gripper", gripper });
            estimate.links.push_back({ arm, "base_link", base });
            estimate.links.push_back({ arm, "wrist", wrist });
            estimate.links.push_back({ arm, "elbow", elbow });
            estimate.joint_angles[arm] = { { "shoulder", 0.4 }, { "forearm", -0.2 },
                                            { "grip", 0.1 } };
        }
        estimate.joints_locked_to_feedback = true;
        pair.frame_id = "camera/depth_optical";
        pair.timing.capture = rbt::time::Timestamp { clock_id(), 1'000'000'000 };
        pair.timing.capture_uncertainty_ns = 1234U;
        // The color stamp deliberately differs from the depth pose timestamp.
        pair.color_capture = { clock_id(), 999'999'000 };
    }

    static rbt_m3t::ModelJoint joint(std::string name, std::string parent, std::string child) {
        rbt_m3t::ModelJoint result;
        result.name = std::move(name);
        result.type = "revolute";
        result.parent = std::move(parent);
        result.child = std::move(child);
        return result;
    }

    const Eigen::Affine3f& link(const std::string& arm, const std::string& name) const {
        for (const auto& entry : estimate.links) {
            if (entry.arm == arm && entry.link == name) {
                return entry.in_camera;
            }
        }
        throw std::runtime_error("missing fixture link");
    }

    void rename_link(const std::string& old_name, const std::string& new_name) {
        if (model.root_link == old_name) model.root_link = new_name;
        for (auto& entry : model.links) if (entry.name == old_name) entry.name = new_name;
        for (auto& joint : model.joints) {
            if (joint.parent == old_name) joint.parent = new_name;
            if (joint.child == old_name) joint.child = new_name;
        }
        for (auto& entry : estimate.links) if (entry.link == old_name) entry.link = new_name;
    }

    void rename_arm(const std::string& old_name, const std::string& new_name) {
        estimate.joint_angles.emplace(new_name, std::move(estimate.joint_angles.at(old_name)));
        estimate.joint_angles.erase(old_name);
        for (auto& entry : estimate.links) if (entry.arm == old_name) entry.arm = new_name;
        for (auto& entry : estimate.mount_refinement) if (entry.arm == old_name) entry.arm = new_name;
    }

    const rbt_m3t::MountRefinementEstimate& mount(const std::string& arm) const {
        for (const auto& entry : estimate.mount_refinement) {
            if (entry.arm == arm) return entry;
        }
        throw std::runtime_error("missing fixture mount");
    }

    std::vector<rbt::tf::DynamicRecord> records(std::string_view prefix = "") const {
        return rbt_m3t::tracking_tf_records(model, estimate, pair, now, prefix, nominal_arms);
    }

    void refresh_mount_diagnostics() {
        for (auto& mount : estimate.mount_refinement) {
            mount.refined_base_in_camera = link(mount.arm, model.root_link);
            const Eigen::Affine3f correction = mount.nominal_base_in_camera.inverse()
                * mount.refined_base_in_camera;
            mount.translation_correction_m = correction.translation().norm();
            mount.rotation_correction_rad = Eigen::AngleAxisf(correction.rotation()).angle();
        }
    }

    void use_encoder_reference(bool refine_mount = false) {
        for (auto mount = estimate.mount_refinement.rbegin(); mount != estimate.mount_refinement.rend(); ++mount) {
            nominal_arms.push_back({ mount->arm, mount->nominal_base_in_camera, {}, false });
        }
        if (!refine_mount) estimate.mount_refinement.clear();
        estimate.joints_locked_to_feedback = false;
        estimate.encoder_reference_links = true;
    }
};

void check_unchanged(const rbt_m3t::TrackingEstimate& actual,
                     const rbt_m3t::TrackingEstimate& before) {
    check(actual.links.size() == before.links.size(), "input link count unchanged");
    for (std::size_t i = 0; i < actual.links.size(); ++i) {
        check(actual.links[i].arm == before.links[i].arm
                  && actual.links[i].link == before.links[i].link
                  && std::memcmp(actual.links[i].in_camera.matrix().data(),
                                 before.links[i].in_camera.matrix().data(), 16U * sizeof(float)) == 0,
              "input names, order and pose bits unchanged");
    }
    check(actual.joint_angles == before.joint_angles && actual.modalities == before.modalities
              && actual.joints_locked_to_feedback == before.joints_locked_to_feedback
              && actual.encoder_reference_links == before.encoder_reference_links
              && actual.visual_link_weights == before.visual_link_weights,
          "input metadata unchanged");
    check(actual.mount_refinement.size() == before.mount_refinement.size(), "mount count unchanged");
    for (std::size_t i = 0; i < actual.mount_refinement.size(); ++i) {
        const auto& a = actual.mount_refinement[i];
        const auto& b = before.mount_refinement[i];
        check(a.arm == b.arm
                  && std::memcmp(a.nominal_base_in_camera.matrix().data(),
                                 b.nominal_base_in_camera.matrix().data(), 16U * sizeof(float)) == 0
                  && std::memcmp(a.refined_base_in_camera.matrix().data(),
                                 b.refined_base_in_camera.matrix().data(), 16U * sizeof(float)) == 0
                  && std::memcmp(&a.translation_correction_m, &b.translation_correction_m, sizeof(double)) == 0
                  && std::memcmp(&a.rotation_correction_rad, &b.rotation_correction_rad, sizeof(double)) == 0,
              "mount metadata bits and order unchanged");
    }
}

void test_complete_native_trees_and_wire_round_trip() {
    Fixture fixture;
    const auto before = fixture.estimate;
    const auto records = fixture.records();
    check(std::string_view(rbt_m3t::TrackingTfCapability::TYPE_NAME) == "perception/M3tTf",
          "native TF capability name");
    check(records.size() == 10U, "two complete trees with mount anchors and no invented TCP");
    check(records == rbt_m3t::tracking_tf_records(fixture.model, fixture.estimate, fixture.pair, fixture.now),
          "public default prefix is empty");
    rbt::tf::TransformBuffer buffer;
    const auto authority = buffer.bind_provider("m3t", "00000000-0000-4000-8000-000000000001");
    const auto legacy = buffer.bind_provider("legacy", "00000000-0000-4000-8000-000000000002");
    rbt::tf::StaticSet statics;
    statics.revision = 1U;
    statics.effective_from = *fixture.pair.timing.capture;
    statics.effective_uncertainty_ns = 0U;
    for (const std::string arm : { "so101_left", "so101_right" }) {
        statics.transforms.push_back({ rbt::tf::FrameId(fixture.pair.frame_id),
                                       rbt::tf::FrameId(arm == "so101_left" ? "left_base" : "right_base"),
                                       rbt::tf::Transform3::identity() });
    }
    buffer.apply_static_snapshot(legacy, std::move(statics));
    std::map<std::string, std::string> parents;
    for (const auto& record : records) {
        check(record.stamp.clock == fixture.pair.timing.capture->clock
                  && record.stamp.ns == fixture.pair.timing.capture->ns
                  && record.stamp.ns != fixture.now.ns
                  && record.uncertainty_ns == fixture.pair.timing.capture_uncertainty_ns
                  && record.transform_epoch == 0U,
              "capture clock, stamp, uncertainty and zero epoch preserved on every edge");
        check(record.child.str().starts_with("so101_"), "all children use canonical arm names");
        check(parents.emplace(std::string(record.child.str()), std::string(record.parent.str())).second,
              "every child appears once");
        const auto wire = rbt::tf::encode_dynamic(authority.physical(), record);
        const auto decoded = rbt::tf::decode_dynamic(authority.physical(), wire);
        check(decoded.parent == record.parent && decoded.child == record.child
                  && decoded.stamp.clock == record.stamp.clock && decoded.stamp.ns == record.stamp.ns
                  && decoded.uncertainty_ns == record.uncertainty_ns && decoded.transform_epoch == 0U,
              "wire round trip preserves identity and timing");
        check_pose(decoded.transform, eigen_pose(record.transform), "wire pose round trip", 1.0e-12);
        buffer.ingest_dynamic(authority, decoded);
    }
    rbt::tf::LookupOptions options;
    options.timestamp_uncertainty_ns = 0U;
    options.now = &fixture.now;
    for (const std::string arm : { "so101_left", "so101_right" }) {
        const auto root = arm + "/base_link";
        const auto anchor = arm + "/cad_mount";
        check(parents.at(anchor) == fixture.pair.frame_id, "camera parents nominal CAD mount");
        check(parents.at(root) == anchor, "CAD mount parents corrected root");
        check((fixture.mount(arm).nominal_base_in_camera.matrix() - Eigen::Matrix4f::Identity())
                  .cwiseAbs().maxCoeff() > 0.1F, "nominal mount is nonidentity");
        check(!(fixture.mount(arm).nominal_base_in_camera.inverse()
                * fixture.link(arm, "base_link")).matrix().isApprox(Eigen::Matrix4f::Identity()),
              "mount correction is nonidentity");
        check(parents.at(arm + "/elbow") == root
                  && parents.at(arm + "/wrist") == arm + "/elbow"
                  && parents.at(arm + "/gripper") == arm + "/wrist",
              "model parent-child tree preserved");
        for (const auto& link : fixture.model.links) {
            const auto actual = buffer.lookup_at(fixture.pair.frame_id,
                arm + '/' + link.name, *fixture.pair.timing.capture, options);
            check_pose(actual.transform, fixture.link(arm, link.name).cast<double>(),
                       "composed camera-to-link equals original estimate");
        }
        const auto relative = buffer.lookup_at(root, arm + "/gripper",
                                               *fixture.pair.timing.capture, options);
        const Eigen::Affine3d expected = fixture.link(arm, "base_link").cast<double>().inverse()
                                        * fixture.link(arm, "gripper").cast<double>();
        check_pose(relative.transform, expected, "root-to-gripper composes through rotated parents");
        check_pose(buffer.lookup_at(fixture.pair.frame_id,
                                    arm == "so101_left" ? "left_base" : "right_base",
                                    *fixture.pair.timing.capture, options).transform,
                   Eigen::Affine3d::Identity(), "legacy static base coexists with canonical dynamic tree");
    }
    // Translation is copied exactly: only the rotation quaternion is normalized.
    for (const auto& record : records) {
        Eigen::Affine3d expected = Eigen::Affine3d::Identity();
        for (const auto& mount : fixture.estimate.mount_refinement) {
            if (record.child.str() == mount.arm + "/cad_mount") {
                expected = mount.nominal_base_in_camera.cast<double>();
            }
        }
        for (const auto& entry : fixture.estimate.links) {
            if (record.child.str() == entry.arm + '/' + entry.link) {
                expected = entry.in_camera.cast<double>();
                if (entry.link == fixture.model.root_link) {
                    expected = fixture.mount(entry.arm).nominal_base_in_camera.cast<double>().inverse()
                               * entry.in_camera.cast<double>();
                } else {
                    for (const auto& joint : fixture.model.joints) {
                        if (joint.child == entry.link) {
                            expected = fixture.link(entry.arm, joint.parent).cast<double>().inverse()
                                       * entry.in_camera.cast<double>();
                        }
                    }
                }
            }
        }
        check_pose(record.transform, expected, "each nominal, correction and model edge is correct");
        const auto translation = record.transform.translation();
        check(translation.x() == expected.translation().x()
                  && translation.y() == expected.translation().y()
                  && translation.z() == expected.translation().z(), "translation is exact");
    }
    check_unchanged(fixture.estimate, before);
}

void test_consecutive_capture_keeps_epoch_zero() {
    Fixture fixture;
    rbt::tf::TransformBuffer buffer;
    const auto authority = buffer.bind_provider("m3t", "00000000-0000-4000-8000-000000000003");
    for (const auto& record : fixture.records()) {
        buffer.ingest_dynamic(authority, record);
    }
    const auto first_stamp = *fixture.pair.timing.capture;
    fixture.pair.timing.capture->ns += 20'000'000;
    for (auto& link : fixture.estimate.links) {
        link.in_camera.translation().x() += 0.02F;
    }
    fixture.refresh_mount_diagnostics();
    for (const auto& record : fixture.records()) {
        check(record.transform_epoch == 0U && record.stamp.ns == first_stamp.ns + 20'000'000,
              "new capture changes stamp, never epoch");
        buffer.ingest_dynamic(authority,
            rbt::tf::decode_dynamic(authority.physical(),
                                   rbt::tf::encode_dynamic(authority.physical(), record)));
    }
    rbt::tf::LookupOptions options;
    options.timestamp_uncertainty_ns = 0U;
    options.now = &fixture.now;
    const auto midpoint = buffer.lookup_at(fixture.pair.frame_id, "so101_left/base_link",
        { first_stamp.clock, first_stamp.ns + 10'000'000 }, options);
    const auto current = buffer.lookup_at(fixture.pair.frame_id, "so101_left/gripper",
                                          *fixture.pair.timing.capture, options);
    check_pose(current.transform, fixture.link("so101_left", "gripper").cast<double>(),
               "new capture resolves in same activation");
    for (const auto& source : midpoint.sources) {
        check(source.transform_epoch == std::optional<std::uint64_t>(0U),
              "interpolation spans consecutive samples in epoch zero");
    }
}

void test_identity_correction_and_consecutive_reset() {
    Fixture fixture;
    rbt::tf::TransformBuffer buffer;
    const auto authority = buffer.bind_provider("m3t", "00000000-0000-4000-8000-000000000004");
    const auto corrected_stamp = *fixture.pair.timing.capture;
    const auto corrected = fixture.records();
    for (const auto& record : corrected) buffer.ingest_dynamic(authority, record);
    std::map<std::string, Eigen::Affine3d> corrected_grippers;
    std::map<std::string, Eigen::Affine3d> reset_grippers;
    std::map<std::string, Eigen::Affine3d> midpoint_grippers;
    for (auto& mount : fixture.estimate.mount_refinement) {
        const Eigen::Affine3d nominal = mount.nominal_base_in_camera.cast<double>();
        const Eigen::Affine3d root = fixture.link(mount.arm, "base_link").cast<double>();
        const Eigen::Affine3d root_to_gripper = root.inverse()
            * fixture.link(mount.arm, "gripper").cast<double>();
        corrected_grippers.emplace(mount.arm, fixture.link(mount.arm, "gripper").cast<double>());
        reset_grippers.emplace(mount.arm, nominal * root_to_gripper);
        // Only the correction changes: independently interpolate its translation
        // and rotation, then compose the complete camera-to-gripper path.
        const Eigen::Affine3d correction = nominal.inverse() * root;
        Eigen::Affine3d midpoint_correction = Eigen::Affine3d::Identity();
        midpoint_correction.translation() = 0.5 * correction.translation();
        Eigen::Quaterniond rotation(correction.linear());
        rotation.normalize();
        midpoint_correction.linear() = rotation.slerp(0.5, Eigen::Quaterniond::Identity()).toRotationMatrix();
        midpoint_grippers.emplace(mount.arm, nominal * midpoint_correction * root_to_gripper);
        // Simulate resetting mount refinement while retaining feedback joint poses.
        for (auto& link : fixture.estimate.links) {
            if (link.arm == mount.arm) {
                link.in_camera = (nominal * root.inverse() * link.in_camera.cast<double>()).cast<float>();
                if (link.link == "base_link") link.in_camera = mount.nominal_base_in_camera;
            }
        }
        mount.refined_base_in_camera = mount.nominal_base_in_camera;
        mount.translation_correction_m = 0.0;
        mount.rotation_correction_rad = 0.0;
    }
    fixture.pair.timing.capture->ns += 20'000'000;
    const auto reset = fixture.records();
    check(reset.size() == corrected.size(), "reset retains every sampled edge");
    std::size_t identity_corrections = 0U;
    for (std::size_t i = 0; i < reset.size(); ++i) {
        const auto& record = reset[i];
        check(record.parent == corrected[i].parent && record.child == corrected[i].child
                  && record.transform_epoch == 0U && corrected[i].transform_epoch == 0U
                  && record.stamp.ns == corrected_stamp.ns + 20'000'000
                  && record.stamp.clock == corrected_stamp.clock
                  && record.uncertainty_ns == corrected[i].uncertainty_ns,
              "corrected-to-reset preserves topology, clock, uncertainty and epoch");
        if (record.parent.str().ends_with("/cad_mount")) {
            check_pose(record.transform, Eigen::Affine3d::Identity(), "reset correction is identity");
            ++identity_corrections;
        } else {
            check_pose(record.transform, eigen_pose(corrected[i].transform),
                       "nominal and relative model edges stay constant across reset");
        }
        buffer.ingest_dynamic(authority, record);
    }
    check(identity_corrections == 2U, "both arms have explicit identity correction samples");
    rbt::tf::LookupOptions options;
    options.timestamp_uncertainty_ns = 0U;
    options.now = &fixture.now;
    for (const auto& [arm, expected] : reset_grippers) {
        check_pose(buffer.lookup_at(fixture.pair.frame_id, arm + "/gripper", corrected_stamp, options).transform,
                   corrected_grippers.at(arm), "corrected capture path remains queryable after reset");
        check_pose(buffer.lookup_at(fixture.pair.frame_id, arm + "/gripper",
                                    *fixture.pair.timing.capture, options).transform,
                   expected, "reset capture camera-to-gripper equals nominal FK");
        const auto midpoint = buffer.lookup_at(fixture.pair.frame_id, arm + "/gripper",
            { corrected_stamp.clock, corrected_stamp.ns + 10'000'000 }, options);
        check_pose(midpoint.transform, midpoint_grippers.at(arm),
                   "corrected-to-reset midpoint composes the interpolated correction");
        for (const auto& source : midpoint.sources) {
            check(source.transform_epoch == std::optional<std::uint64_t>(0U),
                  "corrected-to-reset path interpolates within epoch zero");
        }
    }
    // Identity correction is also valid without any earlier publisher state.
    rbt::tf::TransformBuffer fresh;
    const auto fresh_authority = fresh.bind_provider("m3t", "00000000-0000-4000-8000-000000000005");
    for (const auto& record : fixture.records()) fresh.ingest_dynamic(fresh_authority, record);
    for (const auto& [arm, expected] : reset_grippers) {
        check_pose(fresh.lookup_at(fixture.pair.frame_id, arm + "/gripper",
                                   *fixture.pair.timing.capture, options).transform,
                   expected, "initial nominal pose publishes identity correction and complete path");
    }
}

void test_capture_freshness_and_uncertainty() {
    Fixture fixture;
    fixture.pair.timing.capture->ns = fixture.now.ns - 500'000'000;
    check(!fixture.records().empty(), "500 ms inclusive boundary accepted");
    --fixture.pair.timing.capture->ns;
    check(fixture.records().empty(), "older than 500 ms rejected");
    fixture.pair.timing.capture->ns = fixture.now.ns + 1;
    check(fixture.records().empty(), "future capture rejected");
    fixture.pair.timing.capture->ns = fixture.now.ns;
    check(!fixture.records().empty(), "capture at now accepted");
    for (const auto uncertainty : { std::optional<std::uint64_t> {},
                                     std::optional<std::uint64_t> { 0U } }) {
        fixture.pair.timing.capture_uncertainty_ns = uncertainty;
        for (const auto& record : fixture.records()) {
            check(record.uncertainty_ns == uncertainty, "unknown and zero uncertainty preserved");
        }
    }
    for (const auto& mismatch : { clock_id(2U), clock_id(1U, 4U),
             clock_id(1U, 3U, rbt::time::ClockKind::Device), rbt::time::ClockIdentity {} }) {
        fixture.pair.timing.capture->clock = mismatch;
        check(fixture.records().empty(), "source/generation/domain/unknown clock rejected");
    }
    fixture.pair.timing.capture->clock = clock_id();
    fixture.now.clock = {};
    check(fixture.records().empty(), "unknown now clock rejected");
    fixture.now.clock = clock_id();
    fixture.pair.timing.capture->ns = std::numeric_limits<std::int64_t>::min();
    fixture.now.ns = std::numeric_limits<std::int64_t>::max();
    check(fixture.records().empty(), "age across int64 limits safely rejected");
    fixture.now.clock = clock_id(1U, 3U, rbt::time::ClockKind::Device);
    fixture.now.ns = 1'100'000'000;
    fixture.pair.timing.capture = rbt::time::Timestamp { fixture.now.clock, 1'000'000'000 };
    check(!fixture.records().empty(), "compatible device capture clock accepted");
    for (const auto& record : fixture.records()) {
        check(record.stamp.clock == fixture.now.clock, "complete device clock identity preserved");
    }
    fixture.pair.timing.capture.reset();
    check(fixture.records().empty(), "missing capture suppresses nominal, correction and model edges");
    fixture.estimate.mount_refinement.clear();
    fixture.estimate.joints_locked_to_feedback = false;
    check(fixture.records().empty(), "missing capture returns no records even before metadata validation");
    fixture.now.clock = clock_id();
    fixture.pair.timing.capture = rbt::time::Timestamp { fixture.now.clock, fixture.now.ns - 500'000'001 };
    check(fixture.records().empty(), "stale capture suppresses entire tree before metadata validation");
}

void test_encoder_reference_publishes_fused_links(bool refine_mount) {
    Fixture fixture;
    fixture.use_encoder_reference(refine_mount);
    if (refine_mount) {
        for (auto& nominal : fixture.nominal_arms) {
            nominal.base_in_camera.translation().x() += 0.13F;
        }
    }
    for (auto& joint : fixture.model.joints) {
        joint.lower = -1.0;
        joint.upper = 1.0;
        joint.origin = pose(0.04F, 0.03F, 0.12F, 0.1F, -0.2F, 0.3F);
    }
    std::map<std::string, Eigen::Affine3f> fixed_roots;
    for (const auto& nominal : fixture.nominal_arms) {
        fixed_roots.emplace(nominal.name, fixture.link(nominal.name, fixture.model.root_link));
    }

    rbt::tf::TransformBuffer buffer;
    const auto authority = buffer.bind_provider("m3t", "00000000-0000-4000-8000-000000000007");
    rbt::tf::LookupOptions options;
    options.timestamp_uncertainty_ns = 0U;
    options.now = &fixture.now;
    std::map<std::string, Eigen::Affine3d> first_grippers;

    for (const double visual_shoulder : { 0.2, 0.4 }) {
        fixture.estimate.links.clear();
        fixture.estimate.joint_prior.priors.clear();
        std::map<std::string, Eigen::Affine3f> encoder_grippers;
        for (auto& [arm, angles] : fixture.estimate.joint_angles) {
            angles.at("shoulder") = visual_shoulder;
            auto base = fixed_roots.at(arm);
            if (refine_mount) {
                base.translation().y() += static_cast<float>(0.1 * visual_shoulder);
                base.rotate(Eigen::AngleAxisf(static_cast<float>(visual_shoulder * 0.1),
                                             Eigen::Vector3f::UnitZ()));
            }
            const rbt_m3t::ArmSeed fused { arm, base, angles, !refine_mount };
            for (const auto& link : rbt_m3t::forward_kinematics(fixture.model, fused)) {
                fixture.estimate.links.push_back(link);
            }
            auto measured = fused;
            measured.joint_angles.at("shoulder") = -0.3;
            for (const auto& [joint, angle] : measured.joint_angles) {
                fixture.estimate.joint_prior.priors.push_back({ arm, joint, angle, 0.1 });
            }
            for (const auto& link : rbt_m3t::forward_kinematics(fixture.model, measured)) {
                if (link.link == "gripper") encoder_grippers.emplace(arm, link.in_camera);
            }
        }
        fixture.refresh_mount_diagnostics();
        const auto estimate_before = fixture.estimate;
        const auto records = fixture.records();
        check(records.size() == 10U, "encoder-reference estimate publishes two complete model trees");
        std::map<std::string, std::string> parents;
        for (const auto& record : records) {
            check(record.stamp.clock == fixture.pair.timing.capture->clock
                      && record.stamp.ns == fixture.pair.timing.capture->ns
                      && record.uncertainty_ns == fixture.pair.timing.capture_uncertainty_ns
                      && record.transform_epoch == 0U,
                  "encoder-reference edges retain capture time, uncertainty and epoch");
            check(parents.emplace(record.child.str(), record.parent.str()).second,
                  "each encoder-reference child appears once");
            buffer.ingest_dynamic(authority, record);
        }
        for (const auto& nominal : fixture.nominal_arms) {
            const auto lookup = [&](const std::string& parent, const std::string& child) {
                return buffer.lookup_at(parent, child, *fixture.pair.timing.capture, options).transform;
            };
            const auto cad = nominal.name + "/cad_mount";
            const auto root = nominal.name + "/base_link";
            const auto gripper = nominal.name + "/gripper";
            check(parents.at(cad) == fixture.pair.frame_id && parents.at(root) == cad,
                  "the physical base is parented through the CAD anchor");
            for (const auto& joint : fixture.model.joints) {
                check(parents.at(nominal.name + '/' + joint.child) == nominal.name + '/' + joint.parent,
                      "encoder reference publishes only the canonical model chain");
            }
            const Eigen::Affine3d nominal_pose = nominal.base_in_camera.cast<double>();
            const Eigen::Affine3d root_pose = fixture.link(nominal.name, "base_link").cast<double>();
            const Eigen::Affine3d correction = nominal_pose.inverse() * root_pose;
            check(!correction.matrix().isApprox(Eigen::Matrix4d::Identity()),
                  "fixture has a distinct fixed base and CAD anchor");
            check_pose(lookup(fixture.pair.frame_id, cad), nominal_pose, "configured CAD anchor retained");
            check_pose(lookup(cad, root), correction, "nonidentity calibrated base correction retained");
            check_pose(lookup(fixture.pair.frame_id, root), root_pose, "physical base matches current estimate");
            if (refine_mount) {
                check(!root_pose.matrix().isApprox(fixed_roots.at(nominal.name).matrix().cast<double>(),
                                                  1.0e-3),
                      "soft base moves relative to initialization");
                check(!nominal_pose.matrix().isApprox(
                          fixture.mount(nominal.name).nominal_base_in_camera.matrix().cast<double>(),
                          1.0e-3), "TF CAD anchor is independent of the optimizer prior center");
                check_pose(lookup(cad, root), nominal_pose.inverse() * root_pose,
                           "live correction uses TF CAD anchor and current refined base");
            } else {
                check_pose(lookup(fixture.pair.frame_id, root), fixed_roots.at(nominal.name).cast<double>(),
                           "legacy encoder-reference physical base remains fixed");
            }
            const auto actual = lookup(fixture.pair.frame_id, gripper);
            check_pose(actual, fixture.link(nominal.name, "gripper").cast<double>(),
                       "gripper follows visual fused joint angles");
            check(!eigen_pose(actual).matrix().isApprox(encoder_grippers.at(nominal.name).matrix().cast<double>(),
                                                       1.0e-3),
                  "gripper differs from the encoder-only pose");
            const auto [first, inserted] = first_grippers.emplace(nominal.name, eigen_pose(actual));
            if (!inserted) {
                check(!first->second.matrix().isApprox(eigen_pose(actual).matrix(), 1.0e-3),
                      "visual joint motion changes the gripper");
            }
        }
        check_unchanged(fixture.estimate, estimate_before);
        fixture.pair.timing.capture->ns += 20'000'000;
    }
    fixture.pair.timing.capture->ns = fixture.now.ns - 500'000'001;
    check(fixture.records().empty(), "stale encoder-reference capture suppresses every TF edge");
}

void expect_invalid(const std::function<void(Fixture&)>& mutate, const std::string& message) {
    Fixture fixture;
    auto result = fixture.records();
    const auto original_result = result;
    mutate(fixture);
    const auto before = fixture.estimate;
    bool rejected = false;
    try {
        result = fixture.records();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, message);
    check(result == original_result, "invalid call yields no partial result");
    check_unchanged(fixture.estimate, before);
}

void test_malformed_graphs_and_estimates() {
    expect_invalid([](Fixture& f) { f.model.links.clear(); }, "empty model");
    expect_invalid([](Fixture& f) { f.model.root_link = "missing"; }, "missing root");
    expect_invalid([](Fixture& f) { f.model.links.push_back(f.model.links.front()); }, "duplicate link");
    expect_invalid([](Fixture& f) { f.model.joints.push_back(f.model.joints.front()); }, "duplicate joint");
    expect_invalid([](Fixture& f) { f.model.joints.front().parent = "unknown"; }, "unknown joint parent");
    expect_invalid([](Fixture& f) { f.model.joints.front().child = "unknown"; }, "unknown joint child");
    expect_invalid([](Fixture& f) { f.model.joints.front().parent = "gripper"; }, "self cycle");
    expect_invalid([](Fixture& f) { f.model.joints.front().child = "base_link"; }, "root with parent");
    expect_invalid([](Fixture& f) { f.model.joints.front().child = "elbow"; }, "multiple parents");
    expect_invalid([](Fixture& f) { f.model.joints.pop_back(); }, "disconnected model");
    expect_invalid([](Fixture& f) {
        f.model.joints = { Fixture::joint("a", "wrist", "elbow"),
                           Fixture::joint("b", "elbow", "wrist"),
                           Fixture::joint("c", "base_link", "gripper") };
    }, "disconnected cycle with valid edge count");
    expect_invalid([](Fixture& f) { f.model.joints.front().origin.linear()(0, 0) = -1.0F; },
                   "invalid model origin reflection");
    expect_invalid([](Fixture& f) { f.estimate.links.pop_back(); }, "missing arm link");
    expect_invalid([](Fixture& f) { f.estimate.links.push_back(f.estimate.links.back()); }, "duplicate pose");
    expect_invalid([](Fixture& f) { f.estimate.links.back().link = "unknown"; }, "unknown pose link");
    expect_invalid([](Fixture& f) { f.estimate.links.back().arm = "unknown"; }, "unknown pose arm");
    expect_invalid([](Fixture& f) { f.estimate.joint_angles.clear(); }, "empty arm set");
    expect_invalid([](Fixture& f) { f.estimate.joint_angles.erase("so101_right"); }, "arm sets differ");
    expect_invalid([](Fixture& f) { f.estimate.joint_angles["third"] = {}; }, "too many arms");
    expect_invalid([](Fixture& f) {
        f.estimate.joint_angles[""] = std::move(f.estimate.joint_angles["so101_left"]);
        f.estimate.joint_angles.erase("so101_left");
    }, "empty arm name");
    expect_invalid([](Fixture& f) { f.estimate.links.back().in_camera.linear().col(0) *= -1.0F; },
                   "reflection in last pose rejects entire frame");
    expect_invalid([](Fixture& f) { f.estimate.links.back().in_camera.linear() *= 1.01F; }, "scaled pose");
    expect_invalid([](Fixture& f) { f.estimate.links.back().in_camera.linear()(0, 1) += 0.01F; }, "sheared pose");
    expect_invalid([](Fixture& f) {
        f.estimate.links.back().in_camera.translation().x() = std::numeric_limits<float>::quiet_NaN();
    }, "NaN pose");
    expect_invalid([](Fixture& f) {
        f.estimate.links.back().in_camera.linear()(0, 0) = std::numeric_limits<float>::infinity();
    }, "infinite pose");
    expect_invalid([](Fixture& f) { f.estimate.links.back().in_camera.matrix()(3, 0) = 0.1F; },
                   "invalid homogeneous row");
    expect_invalid([](Fixture& f) { f.pair.frame_id = "/camera"; }, "invalid camera name");
    expect_invalid([](Fixture& f) { f.pair.frame_id = "so101_left/gripper"; }, "camera frame collision");
    expect_invalid([](Fixture& f) { f.model.links.front().name = "bad name"; }, "invalid model name");
}

void test_malformed_mount_metadata_and_frame_collisions() {
    expect_invalid([](Fixture& f) { f.estimate.joints_locked_to_feedback = false; }, "unlocked estimate");
    expect_invalid([](Fixture& f) { f.estimate.mount_refinement.clear(); }, "absent mounts");
    expect_invalid([](Fixture& f) { f.estimate.mount_refinement.pop_back(); }, "one absent mount");
    expect_invalid([](Fixture& f) {
        f.estimate.mount_refinement.back() = f.estimate.mount_refinement.front();
    }, "duplicate mount replaces required arm");
    expect_invalid([](Fixture& f) {
        f.estimate.mount_refinement.push_back(f.estimate.mount_refinement.front());
    }, "extra duplicate mount");
    expect_invalid([](Fixture& f) { f.estimate.mount_refinement.back().arm = "unknown"; }, "unknown mount arm");
    expect_invalid([](Fixture& f) {
        f.estimate.mount_refinement.back().refined_base_in_camera.translation().x() += 0.001F;
    }, "refined translation disagrees with actual root");
    expect_invalid([](Fixture& f) {
        f.estimate.mount_refinement.back().refined_base_in_camera.rotate(
            Eigen::AngleAxisf(0.001F, Eigen::Vector3f::UnitZ()));
    }, "refined rotation disagrees with actual root");
    for (const bool nominal : { true, false }) {
        for (const auto& corrupt : std::vector<std::function<void(Eigen::Affine3f&)>> {
            [](Eigen::Affine3f& p) { p.linear().col(0) *= -1.0F; },
            [](Eigen::Affine3f& p) { p.linear() *= 1.01F; },
            [](Eigen::Affine3f& p) { p.linear()(0, 1) += 0.01F; },
            [](Eigen::Affine3f& p) { p.translation().x() = std::numeric_limits<float>::quiet_NaN(); },
            [](Eigen::Affine3f& p) { p.translation().x() = std::numeric_limits<float>::infinity(); },
            [](Eigen::Affine3f& p) { p.matrix()(3, 0) = 0.1F; } }) {
            expect_invalid([&](Fixture& f) {
                auto& mount = f.estimate.mount_refinement.back();
                corrupt(nominal ? mount.nominal_base_in_camera : mount.refined_base_in_camera);
            }, nominal ? "nonrigid nominal mount" : "nonrigid refined mount");
        }
    }
    expect_invalid([](Fixture& f) {
        f.estimate.mount_refinement.back().translation_correction_m = std::numeric_limits<double>::quiet_NaN();
    }, "NaN correction diagnostic");
    expect_invalid([](Fixture& f) {
        f.estimate.mount_refinement.back().rotation_correction_rad = std::numeric_limits<double>::infinity();
    }, "infinite correction diagnostic");
    expect_invalid([](Fixture& f) {
        f.estimate.mount_refinement.back().translation_correction_m = -1.0;
    }, "negative correction diagnostic");
    expect_invalid([](Fixture& f) {
        f.estimate.mount_refinement.back().translation_correction_m += 0.01;
    }, "incorrect translation correction magnitude");
    expect_invalid([](Fixture& f) {
        f.estimate.mount_refinement.back().rotation_correction_rad += 0.01;
    }, "incorrect rotation correction magnitude");
    expect_invalid([](Fixture& f) { f.pair.frame_id = "so101_left/cad_mount"; }, "camera/mount collision");
    expect_invalid([](Fixture& f) { f.rename_link("elbow", "cad_mount"); }, "model/mount collision");
    expect_invalid([](Fixture& f) { f.rename_link("base_link", "cad_mount"); }, "root/mount collision");
    expect_invalid([](Fixture& f) {
        f.rename_link("elbow", "nested/cad_mount");
        f.rename_arm("so101_right", "so101_left/nested");
    }, "cross-arm model/mount collision");
    expect_invalid([](Fixture& f) {
        f.rename_link("elbow", "nested/gripper");
        f.rename_arm("so101_right", "so101_left/nested");
    }, "cross-arm model/model collision");

    Fixture fixture;
    fixture.estimate.mount_refinement.back().refined_base_in_camera.translation().x() += 1.0e-5F;
    const auto records = fixture.records();
    rbt::tf::TransformBuffer buffer;
    const auto authority = buffer.bind_provider("m3t", "00000000-0000-4000-8000-000000000006");
    for (const auto& record : records) buffer.ingest_dynamic(authority, record);
    rbt::tf::LookupOptions options;
    options.timestamp_uncertainty_ns = 0U;
    options.now = &fixture.now;
    check_pose(buffer.lookup_at(fixture.pair.frame_id, "so101_right/base_link",
                                *fixture.pair.timing.capture, options).transform,
               fixture.link("so101_right", "base_link").cast<double>(),
               "float diagnostic tolerance accepted but output uses actual root", 1.0e-6);
}

void test_malformed_encoder_reference_anchors() {
    expect_invalid([](Fixture& f) {
        f.use_encoder_reference();
        f.estimate.joints_locked_to_feedback = true;
    }, "encoder reference requires visually optimized joints");
    expect_invalid([](Fixture& f) {
        f.estimate.encoder_reference_links = true;
        f.estimate.joints_locked_to_feedback = false;
    }, "combined encoder reference and mount refinement still requires CAD anchors");
    for (const auto& corrupt : std::vector<std::function<void(Fixture&)>> {
        [](Fixture& f) { f.estimate.joints_locked_to_feedback = true; },
        [](Fixture& f) { f.estimate.encoder_reference_links = false; },
        [](Fixture& f) { f.estimate.mount_refinement.pop_back(); },
        [](Fixture& f) { f.estimate.mount_refinement.back().arm = "other"; },
        [](Fixture& f) { f.estimate.mount_refinement.back() = f.estimate.mount_refinement.front(); },
        [](Fixture& f) { f.estimate.mount_refinement.back().nominal_base_in_camera.linear() *= 1.01F; },
        [](Fixture& f) { f.estimate.mount_refinement.back().refined_base_in_camera.translation().x() += 0.001F; },
        [](Fixture& f) { f.estimate.mount_refinement.back().translation_correction_m = 0.0; },
        [](Fixture& f) { f.estimate.mount_refinement.back().rotation_correction_rad = 0.0; } }) {
        expect_invalid([&](Fixture& f) {
            f.use_encoder_reference(true);
            corrupt(f);
        }, "combined mode rejects contradictory flags or mount diagnostics");
    }
    expect_invalid([](Fixture& f) {
        f.use_encoder_reference();
        f.nominal_arms.pop_back();
    }, "each encoder-reference arm requires a nominal CAD anchor");
    expect_invalid([](Fixture& f) {
        f.use_encoder_reference();
        f.nominal_arms.back().name = f.nominal_arms.front().name;
    }, "duplicate encoder-reference CAD anchor rejected");
    expect_invalid([](Fixture& f) {
        f.use_encoder_reference();
        f.nominal_arms.back().name = "other_arm";
    }, "unknown encoder-reference CAD anchor rejected");
    expect_invalid([](Fixture& f) {
        f.use_encoder_reference();
        f.nominal_arms.back().base_in_camera.linear().col(0) *= -1.0F;
    }, "reflected encoder-reference CAD anchor rejected");
}

void test_single_arm_names_and_float_rounding() {
    Fixture fixture;
    fixture.estimate.joint_angles.erase("so101_right");
    fixture.estimate.links.resize(4U);
    fixture.estimate.mount_refinement.resize(1U);
    check(fixture.records("tracking/native").size() == 5U, "single arm and hierarchical prefix accepted");
    for (const auto& prefix : { std::string("/m3t"), std::string("m3t//tf"),
                               std::string(64U, 'x'), std::string(45U, 'x') }) {
        bool rejected = false;
        try {
            (void)fixture.records(prefix);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "invalid/overlong prefix or final frame rejected");
    }
    // Longest generated frame is prefix/so101_left/base_link: 63 visible bytes.
    const std::string limit_prefix(42U, 'x');
    const auto limit_records = fixture.records(limit_prefix);
    check(limit_records.at(1).child.str().size() == 63U, "63 byte TF name accepted");
    fixture.estimate.links.front().in_camera.linear() *= 1.00001F;
    const auto before = fixture.estimate;
    const auto rounded = fixture.records();
    check(rounded.size() == 5U, "native float drift within 1e-4 accepted");
    for (const auto& record : rounded) {
        const auto q = record.transform.rotation();
        check(std::abs(q.x() * q.x() + q.y() * q.y() + q.z() * q.z() + q.w() * q.w() - 1.0) < 1e-12,
              "output quaternion normalized");
    }
    check_unchanged(fixture.estimate, before);
}

} // namespace

int main() {
    try {
        test_complete_native_trees_and_wire_round_trip();
        test_consecutive_capture_keeps_epoch_zero();
        test_identity_correction_and_consecutive_reset();
        test_capture_freshness_and_uncertainty();
        test_encoder_reference_publishes_fused_links(false);
        test_encoder_reference_publishes_fused_links(true);
        test_malformed_graphs_and_estimates();
        test_malformed_mount_metadata_and_frame_collisions();
        test_malformed_encoder_reference_anchors();
        test_single_arm_names_and_float_rounding();
    } catch (const std::exception& error) {
        std::cerr << "M3T tracking TF test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "M3T tracking TF tests passed\n";
    return 0;
}
