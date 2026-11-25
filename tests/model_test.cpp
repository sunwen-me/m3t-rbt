#include <rbt_m3t/model.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rbt_m3t::detail {
double unwrap_and_clamp_joint_angle(double wrapped, double previous, double lower,
                                   double upper);
} // namespace rbt_m3t::detail

namespace {

class TempDirectory {
  public:
    TempDirectory() {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path()
                / ("rbt_m3t_model_test_" + std::to_string(nonce));
        std::filesystem::create_directories(path_);
    }
    ~TempDirectory() { std::filesystem::remove_all(path_); }
    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

  private:
    std::filesystem::path path_;
};

int failures = 0;

void check(bool condition, const std::string& description) {
    if (!condition) {
        std::cerr << "FAIL: " << description << '\n';
        ++failures;
    }
}

template <typename Function>
void expect_invalid(Function&& function, const std::string& description) {
    try {
        std::forward<Function>(function)();
    } catch (const std::invalid_argument&) {
        return;
    } catch (const std::exception& exception) {
        check(false, description + " (threw the wrong exception: " + exception.what() + ")");
        return;
    }
    check(false, description + " (did not throw)");
}

[[nodiscard]] std::filesystem::path write_model(const TempDirectory& directory,
                                                const std::string& contents) {
    const auto path = directory.path() / "robot.json";
    std::ofstream file { path };
    file << contents;
    file.close();
    if (!file) {
        throw std::runtime_error("could not write model test fixture");
    }
    return path;
}

[[nodiscard]] std::string model_json() {
    return R"({
      "schema_version": 1,
      "name": "arm fixture",
      "root_link": "base",
      "units": "metres_radians",
      "links": [
        {"name": "base", "mesh": "", "visuals": 0},
        {"name": "elbow", "mesh": "", "visuals": 0},
        {"name": "wrist", "mesh": "", "visuals": 0}
      ],
      "joints": [
        {"name": "1", "type": "revolute", "parent": "base", "child": "elbow",
         "origin": [0.955336489, 0, 0.295520207, 1, 0, 1, 0, 2,
                    -0.295520207, 0, 0.955336489, 3, 0, 0, 0, 1],
         "axis": [0, 0, 1], "lower": -1.37, "upper": 4.4906552},
        {"name": "fixed_wrist", "type": "fixed", "parent": "elbow", "child": "wrist",
         "origin": [1, 0, 0, 0.5, 0, 0.980066578, 0.198669331, 0,
                    0, -0.198669331, 0.980066578, 0.25, 0, 0, 0, 1],
         "axis": [0, 0, 0], "lower": 0, "upper": 0}
      ]
    })";
}

[[nodiscard]] rbt_m3t::RectifiedPair rectified_pair() {
    std::array<std::uint8_t, 48> source_id {};
    source_id[0] = 1U;
    const auto clock = rbt::time::ClockIdentity::create(
        rbt::time::ClockKind::Device, rbt::time::ClockSourceId(std::move(source_id)), 0U)
                           .value();
    const auto capture = rbt::time::Timestamp::create(clock, 1'000).value();
    const auto received = rbt::time::Timestamp::create(clock, 2'000).value();
    const auto timing = rbt::time::SensorTiming::create(
                            capture, received, std::nullopt, std::nullopt)
                            .value();
    return rbt_m3t::RectifiedPair {
        cv::Mat::zeros(2, 2, CV_8UC3), cv::Mat::zeros(2, 2, CV_16UC1),
        m3t::Intrinsics { 1.0F, 1.0F, 0.5F, 0.5F, 2, 2 }, 1.0,
        timing, capture, "camera", 0U
    };
}

[[nodiscard]] const rbt_m3t::LinkEstimate& link_estimate(
    const std::vector<rbt_m3t::LinkEstimate>& estimates, const std::string& name) {
    const auto found = std::find_if(estimates.begin(), estimates.end(),
                                    [&name](const auto& estimate) {
                                        return estimate.link == name;
                                    });
    if (found == estimates.end()) {
        throw std::runtime_error("link estimate not found: " + name);
    }
    return *found;
}

[[nodiscard]] const rbt_m3t::LinkEstimate& arm_link_estimate(
    const std::vector<rbt_m3t::LinkEstimate>& estimates,
    const std::string& arm, const std::string& link) {
    const auto found = std::find_if(estimates.begin(), estimates.end(),
                                    [&arm, &link](const auto& estimate) {
                                        return estimate.arm == arm && estimate.link == link;
                                    });
    if (found == estimates.end()) {
        throw std::runtime_error("arm link estimate not found: " + arm + "/" + link);
    }
    return *found;
}

void loads_json_and_resolves_relative_meshes() {
    TempDirectory directory;
    auto json = model_json();
    const auto mesh_position = json.find("\"mesh\": \"\"");
    check(mesh_position != std::string::npos, "fixture has an empty mesh field");
    if (mesh_position == std::string::npos) {
        return;
    }
    json.replace(mesh_position, std::string("\"mesh\": \"\"").size(),
                 "\"mesh\": \"meshes/base.obj\"");

    const auto model = rbt_m3t::load_robot_model(write_model(directory, json));
    check(model.links.size() == 3U, "JSON model contains all links");
    check(model.name == "arm fixture", "JSON model retains its name");
    check(model.root_link == "base", "JSON model retains its root link");
    check(model.links[0].mesh == directory.path() / "meshes/base.obj",
          "relative mesh is resolved against the JSON directory");
    check(model.links[1].mesh.empty(), "empty mesh represents a kinematic link");
    check(model.joints.size() == 2U, "JSON model contains all joints");
    check(model.joints[0].name == "1", "joint names are retained");
    check(model.joints[1].type == "fixed", "fixed joint type is retained");
}

void forward_kinematics_uses_origin_then_z_rotation_and_fixed_frames() {
    TempDirectory directory;
    const auto model = rbt_m3t::load_robot_model(write_model(directory, model_json()));
    rbt_m3t::ArmSeed seed;
    seed.name = "left";
    seed.fixed_base = true;
    seed.base_in_camera = Eigen::Translation3f { 0.2F, -0.4F, 1.1F }
                          * Eigen::AngleAxisf { -0.3F, Eigen::Vector3f::UnitZ() };
    seed.joint_angles = { { "1", 4.0 } };

    const auto estimates = rbt_m3t::forward_kinematics(model, seed);
    check(estimates.size() == 3U, "FK emits every articulated link");

    Eigen::Affine3f origin_one = Eigen::Affine3f::Identity();
    origin_one.linear() = Eigen::AngleAxisf { 0.3F, Eigen::Vector3f::UnitY() }.toRotationMatrix();
    origin_one.translation() = Eigen::Vector3f { 1.0F, 2.0F, 3.0F };
    Eigen::Affine3f origin_fixed = Eigen::Affine3f::Identity();
    origin_fixed.linear() = Eigen::AngleAxisf { -0.2F, Eigen::Vector3f::UnitX() }.toRotationMatrix();
    origin_fixed.translation() = Eigen::Vector3f { 0.5F, 0.0F, 0.25F };
    const Eigen::Affine3f elbow = seed.base_in_camera * origin_one
                                  * Eigen::AngleAxisf { 4.0F, Eigen::Vector3f::UnitZ() };
    const Eigen::Affine3f wrist = elbow * origin_fixed;
    check(link_estimate(estimates, "base").in_camera.matrix().isApprox(
              seed.base_in_camera.matrix(), 1.0e-5F),
          "FK preserves the base pose");
    check(link_estimate(estimates, "elbow").in_camera.matrix().isApprox(
              elbow.matrix(), 1.0e-5F),
          "FK composes joint origin before the nontrivial Z rotation");
    check(link_estimate(estimates, "wrist").in_camera.matrix().isApprox(
              wrist.matrix(), 1.0e-5F),
          "FK composes the fixed child origin");
}

void requires_exactly_every_finite_in_limit_revolute_angle() {
    TempDirectory directory;
    const auto model = rbt_m3t::load_robot_model(write_model(directory, model_json()));
    rbt_m3t::ArmSeed seed;
    seed.name = "right";
    seed.joint_angles = { { "1", 0.0 } };
    check(rbt_m3t::forward_kinematics(model, seed).size() == 3U,
          "one exact valid revolute angle is accepted");

    seed.joint_angles.clear();
    expect_invalid([&] { static_cast<void>(rbt_m3t::forward_kinematics(model, seed)); },
                   "missing revolute angle is rejected");
    seed.joint_angles = { { "1", 0.0 }, { "fixed_wrist", 0.0 } };
    expect_invalid([&] { static_cast<void>(rbt_m3t::forward_kinematics(model, seed)); },
                   "fixed-joint angle is rejected as an extra seed value");
    seed.joint_angles = { { "1", 4.6 } };
    expect_invalid([&] { static_cast<void>(rbt_m3t::forward_kinematics(model, seed)); },
                   "out-of-limit angle is rejected");
    seed.joint_angles = { { "1", std::numeric_limits<double>::quiet_NaN() } };
    expect_invalid([&] { static_cast<void>(rbt_m3t::forward_kinematics(model, seed)); },
                   "non-finite angle is rejected");
    seed.joint_angles = { { "1", 0.0 } };
    seed.base_in_camera.linear()(0, 0) = 2.0F;
    expect_invalid([&] { static_cast<void>(rbt_m3t::forward_kinematics(model, seed)); },
                   "non-rigid base pose is rejected");
}

void rejects_invalid_axes_matrices_and_trees() {
    TempDirectory directory;
    auto json = model_json();
    auto axis_position = json.find("\"axis\": [0, 0, 1]");
    check(axis_position != std::string::npos, "fixture has a revolute axis");
    if (axis_position != std::string::npos) {
        json.replace(axis_position, std::string("\"axis\": [0, 0, 1]").size(),
                     "\"axis\": [1, 0, 0]");
        expect_invalid([&] {
            static_cast<void>(rbt_m3t::load_robot_model(write_model(directory, json)));
        }, "unsupported revolute axis is rejected");
    }

    json = model_json();
    const auto rotation_position = json.find("0.955336489");
    check(rotation_position != std::string::npos, "fixture has an origin rotation");
    if (rotation_position != std::string::npos) {
        json.replace(rotation_position, std::string("0.955336489").size(), "2");
        expect_invalid([&] {
            static_cast<void>(rbt_m3t::load_robot_model(write_model(directory, json)));
        }, "non-rigid joint origin is rejected");
    }

    json = model_json();
    const auto parent_position = json.find("\"parent\": \"elbow\"");
    check(parent_position != std::string::npos, "fixture has a fixed joint parent");
    if (parent_position != std::string::npos) {
        json.replace(parent_position, std::string("\"parent\": \"elbow\"").size(),
                     "\"parent\": \"wrist\"");
        expect_invalid([&] {
            static_cast<void>(rbt_m3t::load_robot_model(write_model(directory, json)));
        }, "cyclic/disconnected model is rejected");
    }
}

void constructor_accepts_one_or_two_arm_seeds_and_rejects_invalid_cardinality() {
    TempDirectory directory;
    const auto model = rbt_m3t::load_robot_model(write_model(directory, model_json()));
    rbt_m3t::ArmSeed left;
    left.name = "left";
    left.joint_angles = { { "1", 0.1 } };
    rbt_m3t::ArmSeed right = left;
    right.name = "right";
    auto images = std::make_shared<rbt_m3t::CameraImages>();
    rbt_m3t::EngineOptions options;
    options.cache_directory = directory.path() / "cache";

    try {
        rbt_m3t::TrackerEngine engine { model, { left }, options, images };
        check(true, "engine accepts one valid arm seed");
    } catch (const std::exception& exception) {
        check(false, "engine accepts one valid arm seed: "
                         + std::string(exception.what()));
    }
    expect_invalid([&] {
        rbt_m3t::TrackerEngine engine { model, {}, options, images };
        static_cast<void>(engine);
    }, "engine rejects an empty arm-seed list");
    try {
        rbt_m3t::TrackerEngine engine { model, { left, right }, options, images };
        check(true, "engine accepts two unique valid arm seeds");
    } catch (const std::exception& exception) {
        check(false, "engine accepts two unique valid arm seeds: "
                         + std::string(exception.what()));
    }
    auto third = left;
    third.name = "third";
    expect_invalid([&] {
        rbt_m3t::TrackerEngine engine { model, { left, right, third }, options, images };
        static_cast<void>(engine);
    }, "engine rejects more than two arm seeds");
    right.name = "left";
    expect_invalid([&] {
        rbt_m3t::TrackerEngine engine { model, { left, right }, options, images };
        static_cast<void>(engine);
    }, "engine rejects duplicate arm names");
    right.name = "right";
    options.model_points = 10'001;
    expect_invalid([&] {
        rbt_m3t::TrackerEngine engine { model, { left, right }, options, images };
        static_cast<void>(engine);
    }, "engine rejects an out-of-bound model point count");
}

void tracker_keeps_fixed_bases_in_world_while_the_camera_moves() {
    TempDirectory directory;
    rbt_m3t::RobotModel model;
    model.name = "virtual fixture";
    model.root_link = "base";
    model.links.push_back(rbt_m3t::ModelLink { "base", {} });
    model.links.push_back(rbt_m3t::ModelLink { "tool", {} });
    rbt_m3t::ModelJoint joint;
    joint.name = "joint";
    joint.type = "revolute";
    joint.parent = "base";
    joint.child = "tool";
    joint.origin.translation() = Eigen::Vector3f { 0.1F, -0.2F, 0.3F };
    joint.lower = -1.0;
    joint.upper = 1.0;
    model.joints.push_back(joint);

    rbt_m3t::ArmSeed left;
    left.name = "left";
    left.fixed_base = true;
    left.joint_angles = { { "joint", 0.25 } };
    left.base_in_camera = Eigen::Translation3f { 0.2F, -0.1F, 0.8F }
                          * Eigen::AngleAxisf { 0.15F, Eigen::Vector3f::UnitY() };
    rbt_m3t::ArmSeed right;
    right.name = "right";
    right.fixed_base = true;
    right.joint_angles = { { "joint", -0.4 } };
    right.base_in_camera = Eigen::Translation3f { -0.3F, 0.4F, 1.1F }
                           * Eigen::AngleAxisf { -0.2F, Eigen::Vector3f::UnitX() };
    const std::vector<rbt_m3t::ArmSeed> seeds { left, right };

    rbt_m3t::EngineOptions options;
    options.cache_directory = directory.path() / "cache";
    options.use_region = false;
    auto images = std::make_shared<rbt_m3t::CameraImages>();
    images->current = rectified_pair();
    const Eigen::Affine3f first_camera_pose =
        Eigen::Translation3f { 0.4F, 0.3F, -0.2F }
        * Eigen::AngleAxisf { 0.3F, Eigen::Vector3f::UnitZ() };
    images->camera_in_world = first_camera_pose;

    auto invalid_images = std::make_shared<rbt_m3t::CameraImages>(*images);
    invalid_images->camera_in_world.linear()(0, 0) = 2.0F;
    auto invalid_options = options;
    invalid_options.cache_directory = directory.path() / "invalid_cache";
    rbt_m3t::TrackerEngine invalid_engine {
        model, seeds, invalid_options, invalid_images
    };
    expect_invalid([&] { invalid_engine.set_up(); },
                   "engine setup rejects an invalid camera pose");
    check(!std::filesystem::exists(invalid_options.cache_directory),
          "invalid camera pose is rejected before setup creates tracker state");

    rbt_m3t::TrackerEngine engine { model, seeds, options, images };
    engine.set_up();
    const auto first = engine.step(0);
    check(arm_link_estimate(first.links, "left", "base").in_camera.matrix().isApprox(
              left.base_in_camera.matrix(), 1.0e-5F),
          "first estimate preserves the left camera-frame seed");
    check(arm_link_estimate(first.links, "left", "base").arm == "left",
          "first base estimate belongs to the left arm");

    const Eigen::Affine3f second_camera_pose =
        Eigen::Translation3f { -0.5F, 0.2F, 0.6F }
        * Eigen::AngleAxisf { -0.25F, Eigen::Vector3f::UnitY() };
    Eigen::Affine3f invalid_camera_pose = second_camera_pose;
    invalid_camera_pose.linear()(0, 0) = 2.0F;
    images->camera_in_world = invalid_camera_pose;
    expect_invalid([&] { static_cast<void>(engine.step(1)); },
                   "engine step rejects an invalid current camera pose");
    expect_invalid([&] { engine.reinitialize(seeds, 1); },
                   "engine reinitialization rejects an invalid current camera pose");

    images->camera_in_world = second_camera_pose;
    const auto second = engine.step(1);
    for (const auto& seed : seeds) {
        const auto initial_fk = rbt_m3t::forward_kinematics(model, seed);
        for (const auto& link : model.links) {
            const auto& estimate = arm_link_estimate(second.links, seed.name, link.name);
            const auto& initial_estimate = arm_link_estimate(first.links, seed.name, link.name);
            const auto expected_camera_pose = second_camera_pose.inverse()
                                              * first_camera_pose
                                              * link_estimate(initial_fk, link.name).in_camera;
            check(estimate.arm == seed.name,
                  "moved-camera estimate retains the arm identity");
            check(estimate.in_camera.matrix().isApprox(
                      expected_camera_pose.matrix(), 1.0e-5F),
                  "fixed articulated pose remains at its world pose as the camera moves");
            const Eigen::Affine3f initial_world_pose =
                first_camera_pose * initial_estimate.in_camera;
            const Eigen::Affine3f moved_world_pose =
                second_camera_pose * estimate.in_camera;
            check(initial_world_pose.matrix().isApprox(moved_world_pose.matrix(), 1.0e-5F),
                  "camera-frame estimates recover the same fixed world pose");
        }
    }
}

void dual_camera_engine_validates_both_views_and_reports_primary_coordinates() {
    TempDirectory directory;
    rbt_m3t::RobotModel model;
    model.name = "dual-view virtual fixture";
    model.root_link = "base";
    model.links.push_back(rbt_m3t::ModelLink { "base", {} });
    model.links.push_back(rbt_m3t::ModelLink { "tool", {} });
    rbt_m3t::ModelJoint joint;
    joint.name = "joint";
    joint.type = "revolute";
    joint.parent = "base";
    joint.child = "tool";
    joint.origin.translation() = Eigen::Vector3f { 0.15F, -0.05F, 0.25F };
    joint.lower = -1.0;
    joint.upper = 1.0;
    model.joints.push_back(joint);

    rbt_m3t::ArmSeed left;
    left.name = "left";
    left.fixed_base = true;
    left.joint_angles = { { "joint", 0.2 } };
    left.base_in_camera = Eigen::Translation3f { 0.2F, -0.1F, 0.8F }
                          * Eigen::AngleAxisf { 0.15F, Eigen::Vector3f::UnitY() };
    rbt_m3t::ArmSeed right = left;
    right.name = "right";
    right.joint_angles["joint"] = -0.35;
    right.base_in_camera = Eigen::Translation3f { -0.3F, 0.25F, 1.0F }
                           * Eigen::AngleAxisf { -0.2F, Eigen::Vector3f::UnitX() };
    const std::vector<rbt_m3t::ArmSeed> seeds { left, right };

    rbt_m3t::EngineOptions options;
    options.cache_directory = directory.path() / "dual_cache";
    options.use_region = false;
    auto primary = std::make_shared<rbt_m3t::CameraImages>();
    primary->current = rectified_pair();
    const Eigen::Affine3f first_primary_pose =
        Eigen::Translation3f { 0.4F, 0.3F, -0.2F }
        * Eigen::AngleAxisf { 0.3F, Eigen::Vector3f::UnitZ() };
    primary->camera_in_world = first_primary_pose;
    auto secondary = std::make_shared<rbt_m3t::CameraImages>();
    secondary->current = rectified_pair();
    const Eigen::Affine3f secondary_pose =
        Eigen::Translation3f { -0.8F, 0.6F, 0.9F }
        * Eigen::AngleAxisf { -0.25F, Eigen::Vector3f::UnitY() };
    secondary->camera_in_world = secondary_pose;
    check(!secondary_pose.matrix().isApprox(first_primary_pose.matrix(), 1.0e-5F),
          "dual-view fixture uses distinct primary and secondary camera poses");

    expect_invalid([&] {
        rbt_m3t::TrackerEngine aliased { model, seeds, options, primary, primary };
        static_cast<void>(aliased);
    }, "dual-view engine rejects aliased primary and secondary image owners");

    auto missing_secondary = std::make_shared<rbt_m3t::CameraImages>();
    missing_secondary->camera_in_world = secondary_pose;
    auto missing_options = options;
    missing_options.cache_directory = directory.path() / "missing_secondary_cache";
    rbt_m3t::TrackerEngine missing_engine {
        model, seeds, missing_options, primary, missing_secondary
    };
    expect_invalid([&] { missing_engine.set_up(); },
                   "dual-view setup rejects a secondary view without a current RGB-D pair");
    check(!std::filesystem::exists(missing_options.cache_directory),
          "missing secondary data is rejected before tracker setup creates cache state");

    auto invalid_secondary_pose = std::make_shared<rbt_m3t::CameraImages>();
    invalid_secondary_pose->current = rectified_pair();
    invalid_secondary_pose->camera_in_world = secondary_pose;
    invalid_secondary_pose->camera_in_world.linear()(0, 0) = 2.0F;
    auto invalid_pose_options = options;
    invalid_pose_options.cache_directory = directory.path() / "invalid_secondary_pose_cache";
    rbt_m3t::TrackerEngine invalid_pose_engine {
        model, seeds, invalid_pose_options, primary, invalid_secondary_pose
    };
    expect_invalid([&] { invalid_pose_engine.set_up(); },
                   "dual-view setup rejects an invalid secondary camera pose");
    check(!std::filesystem::exists(invalid_pose_options.cache_directory),
          "invalid secondary pose is rejected before tracker setup creates cache state");

    rbt_m3t::TrackerEngine engine { model, seeds, options, primary, secondary };
    engine.set_up();
    const auto first = engine.step(0);
    for (const auto& seed : seeds) {
        const auto expected_links = rbt_m3t::forward_kinematics(model, seed);
        for (const auto& expected : expected_links) {
            check(arm_link_estimate(first.links, seed.name, expected.link)
                      .in_camera.matrix().isApprox(expected.in_camera.matrix(), 1.0e-5F),
                  "dual-view estimates stay expressed in the primary camera frame");
        }
    }

    secondary->current.reset();
    expect_invalid([&] { static_cast<void>(engine.step(1)); },
                   "step rejects a missing current secondary RGB-D pair");
    expect_invalid([&] { engine.reinitialize(seeds, 1); },
                   "reinitialization rejects a missing current secondary RGB-D pair");
    secondary->current = rectified_pair();

    secondary->current->depth_scale = 0.0;
    expect_invalid([&] { static_cast<void>(engine.step(1)); },
                   "step rejects malformed current secondary RGB-D data");
    expect_invalid([&] { engine.reinitialize(seeds, 1); },
                   "reinitialization rejects malformed current secondary RGB-D data");
    secondary->current = rectified_pair();

    Eigen::Affine3f malformed_secondary_pose = secondary_pose;
    malformed_secondary_pose.linear()(0, 0) = 2.0F;
    secondary->camera_in_world = malformed_secondary_pose;
    expect_invalid([&] { static_cast<void>(engine.step(1)); },
                   "step rejects an invalid current secondary camera pose");
    expect_invalid([&] { engine.reinitialize(seeds, 1); },
                   "reinitialization rejects an invalid current secondary camera pose");
    secondary->camera_in_world = secondary_pose;

    const Eigen::Affine3f second_primary_pose =
        Eigen::Translation3f { -0.5F, 0.2F, 0.6F }
        * Eigen::AngleAxisf { -0.25F, Eigen::Vector3f::UnitY() };
    primary->camera_in_world = second_primary_pose;
    const auto second = engine.step(1);
    for (const auto& seed : seeds) {
        const auto expected_links = rbt_m3t::forward_kinematics(model, seed);
        for (const auto& expected : expected_links) {
            const auto expected_in_primary = second_primary_pose.inverse()
                                             * first_primary_pose
                                             * expected.in_camera;
            check(arm_link_estimate(second.links, seed.name, expected.link)
                      .in_camera.matrix().isApprox(expected_in_primary.matrix(), 1.0e-5F),
                  "moving the primary camera controls output coordinates with a distinct secondary pose");
        }
    }
}

void reinitialization_validates_priors_before_mutating_poses() {
    TempDirectory directory;
    const auto model = rbt_m3t::load_robot_model(write_model(directory, model_json()));
    rbt_m3t::ArmSeed left;
    left.name = "left";
    left.fixed_base = true;
    left.joint_angles = { { "1", 0.25 } };
    auto right = left;
    right.name = "right";
    right.base_in_camera.translation().x() = 0.5F;
    const std::vector<rbt_m3t::ArmSeed> seeds { left, right };
    rbt_m3t::EngineOptions options;
    options.cache_directory = directory.path() / "cache";
    options.use_region = false;
    options.refine_initialization = true;
    auto images = std::make_shared<rbt_m3t::CameraImages>();
    images->current = rectified_pair();
    rbt_m3t::TrackerEngine engine { model, seeds, options, images };
    engine.set_up();
    const auto original = engine.step(0, { { "left", "1", 0.6, 0.1 } });
    // Virtual links have no image modality, so native Refiner selects no
    // optimizer. The fake-modality test covers actual prior-driven updates.
    check(std::abs(original.joint_angles.at("left").at("1") - 0.25) < 1.0e-5,
          "virtual fixture preserves its seed through startup detection");

    auto requested = seeds;
    requested[0].base_in_camera.translation().x() = -0.75F;
    requested[0].joint_angles["1"] = -0.3;
    requested[1].joint_angles["1"] = 0.4;
    std::reverse(requested.begin(), requested.end());
    const std::vector<std::vector<rbt_m3t::JointAnglePrior>> malformed {
        { { "left", "1", 0.2, 0.0 } },
        { { "left", "1", 0.2, 0.1 }, { "right", "1", 0.2, 0.0 } },
        { { "missing", "1", 0.2, 0.1 } },
        { { "left", "fixed_wrist", 0.2, 0.1 } },
        { { "left", "1", 0.2, 0.1 }, { "left", "1", 0.3, 0.1 } },
        { { "left", "1", 5.0, 0.1 } },
        { { "left", "1", std::numeric_limits<double>::quiet_NaN(), 0.1 } },
        { { "left", "1", 0.2, std::numeric_limits<double>::denorm_min() } },
    };
    int iteration = 1;
    for (const auto& priors : malformed) {
        expect_invalid([&] { engine.reinitialize(requested, iteration, priors); },
                       "malformed reinitialization prior is rejected");
        const auto unchanged = engine.step(iteration++);
        for (const auto& previous : original.links) {
            check(arm_link_estimate(unchanged.links, previous.arm, previous.link)
                      .in_camera.matrix().isApprox(previous.in_camera.matrix(), 1.0e-5F),
                  "malformed reinitialization leaves every arm and link pose unchanged");
        }
    }

    engine.reinitialize(requested, iteration++, { { "right", "1", 0.7, 0.1 } });
    const auto reinitialized = engine.step(iteration++);
    for (const auto& seed : requested) {
        for (const auto& expected : rbt_m3t::forward_kinematics(model, seed)) {
            check(arm_link_estimate(reinitialized.links, seed.name, expected.link)
                      .in_camera.matrix().isApprox(expected.in_camera.matrix(), 1.0e-5F),
                  "reinitialization accepts reordered seeds and clears detection priors");
        }
    }

    // A legacy call after a prior-bearing update must not reuse its residual.
    static_cast<void>(engine.step(iteration++, { { "left", "1", 0.9, 0.1 } }));
    engine.reinitialize(seeds);
    const auto legacy = engine.step(iteration++);
    for (const auto& seed : seeds) {
        for (const auto& expected : rbt_m3t::forward_kinematics(model, seed)) {
            check(arm_link_estimate(legacy.links, seed.name, expected.link)
                      .in_camera.matrix().isApprox(expected.in_camera.matrix(), 1.0e-5F),
                  "legacy reinitialization remains valid without retaining old priors");
        }
    }
}

void mount_refinement_validates_configuration_before_setup() {
    TempDirectory directory;
    const auto model = rbt_m3t::load_robot_model(write_model(directory, model_json()));
    rbt_m3t::ArmSeed left;
    left.name = "left";
    left.joint_angles = { { "1", 0.1 } };
    auto right = left;
    right.name = "right";
    auto images = std::make_shared<rbt_m3t::CameraImages>();
    rbt_m3t::EngineOptions options;
    options.cache_directory = directory.path() / "mount_cache";
    options.mount_refinement.enabled = true;
    const auto rejects = [&](const rbt_m3t::EngineOptions& requested,
                             const std::string& description) {
        expect_invalid([&] {
            rbt_m3t::TrackerEngine engine { model, { left, right }, requested, images };
        }, description);
        check(!std::filesystem::exists(options.cache_directory),
              "invalid mount configuration does not create tracker state");
    };
    for (const double sigma : { 0.0, 0.000099, 0.200001,
             std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::quiet_NaN() }) {
        auto invalid = options;
        invalid.mount_refinement.translation_standard_deviation_m = sigma;
        rejects(invalid, "mount engine rejects out-of-range or non-finite translation sigma");
    }
    for (const double sigma : { -0.1, 0.000099, 0.785399,
             std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::quiet_NaN() }) {
        auto invalid = options;
        invalid.mount_refinement.rotation_standard_deviation_rad = sigma;
        rejects(invalid, "mount engine rejects out-of-range or non-finite rotation sigma");
    }
    for (const bool static_detector : { false, true }) {
        auto invalid = options;
        invalid.use_static_detector = static_detector;
        invalid.refine_initialization = !static_detector;
        rejects(invalid, "mount engine rejects automatic detection and initialization refinement");
    }
    left.fixed_base = true;
    rejects(options, "mount engine rejects a fixed left base");
    left.fixed_base = false;
    right.fixed_base = true;
    rejects(options, "mount engine rejects a fixed right base");
    right.fixed_base = false;
    for (const bool upper : { false, true }) {
        auto boundary = options;
        boundary.mount_refinement.translation_standard_deviation_m = upper ? 0.2 : 0.0001;
        boundary.mount_refinement.rotation_standard_deviation_rad = upper ? 0.785398 : 0.0001;
        rbt_m3t::TrackerEngine engine { model, { left, right }, boundary, images };
    }
    // Disabled mode retains its existing option and fixed-base behavior.
    options.mount_refinement.enabled = false;
    options.mount_refinement.translation_standard_deviation_m = 0.0;
    options.mount_refinement.rotation_standard_deviation_rad = 0.0;
    options.use_static_detector = true;
    options.refine_initialization = true;
    left.fixed_base = true;
    rbt_m3t::TrackerEngine legacy { model, { left, right }, options, images };
}

void mount_refinement_locks_complete_feedback_and_resets_nominal() {
    TempDirectory directory;
    auto model = rbt_m3t::load_robot_model(write_model(directory, model_json()));
    model.links.push_back({ "gripper_link", {} });
    rbt_m3t::ModelJoint gripper;
    gripper.name = "gripper";
    gripper.type = "revolute";
    gripper.parent = "wrist";
    gripper.child = "gripper_link";
    gripper.origin.translation() = Eigen::Vector3f { 0.04F, 0.02F, -0.01F };
    gripper.lower = -0.4;
    gripper.upper = 0.8;
    model.joints.push_back(gripper);
    rbt_m3t::ArmSeed left;
    left.name = "left";
    left.base_in_camera = Eigen::Translation3f { 0.15F, -0.23F, 0.8F }
                          * Eigen::AngleAxisf { 0.37F, Eigen::Vector3f::UnitY() };
    left.joint_angles = { { "1", 0.1 }, { "gripper", 0.2 } };
    auto right = left;
    right.name = "right";
    right.base_in_camera = Eigen::Translation3f { -0.31F, 0.17F, 0.6F }
                           * Eigen::AngleAxisf { -0.29F, Eigen::Vector3f::UnitX() };
    const std::vector<rbt_m3t::ArmSeed> seeds { left, right };
    rbt_m3t::EngineOptions options;
    options.cache_directory = directory.path() / "mount_cache";
    options.use_region = false; // Body-free graph exercises the actual engine without OpenGL.
    options.mount_refinement.enabled = true;
    auto images = std::make_shared<rbt_m3t::CameraImages>();
    images->current = rectified_pair();
    images->camera_in_world = Eigen::Translation3f { 0.4F, -0.3F, 0.2F }
                              * Eigen::AngleAxisf { 0.41F, Eigen::Vector3f::UnitZ() };
    rbt_m3t::TrackerEngine engine { model, seeds, options, images };
    engine.set_up();
    const std::vector<rbt_m3t::JointAnglePrior> feedback {
        { "right", "gripper", 0.123456789012345, 0.05 },
        { "left", "1", 1.23456789012345, 0.05 },
        { "right", "1", -1.23456789012345, 0.05 },
        { "left", "gripper", -0.23456789012345, 0.05 },
    };
    int iteration = 0;
    const auto verify = [&](const rbt_m3t::TrackingEstimate& estimate,
                            const std::vector<rbt_m3t::ArmSeed>& nominal,
                            const std::vector<rbt_m3t::JointAnglePrior>& measured) {
        check(estimate.joints_locked_to_feedback, "mount mode reports locked joints");
        check(estimate.links.size() == nominal.size() * model.links.size(),
              "internal CAD anchors do not leak into public link estimates");
        check(estimate.mount_refinement.size() == nominal.size(),
              "mount mode reports one correction for each arm");
        const auto expected_seeds = rbt_m3t::seeds_with_joint_priors(model, nominal, measured);
        for (const auto& seed : expected_seeds) {
            for (const auto& expected : rbt_m3t::forward_kinematics(model, seed)) {
                check(arm_link_estimate(estimate.links, seed.name, expected.link)
                          .in_camera.matrix().isApprox(expected.in_camera.matrix(), 2.0e-5F),
                      "locked encoder transforms propagate before optimization without doubling CAD poses");
            }
            const auto diagnostic = std::find_if(estimate.mount_refinement.begin(),
                estimate.mount_refinement.end(), [&seed](const auto& item) {
                    return item.arm == seed.name;
                });
            if (diagnostic == estimate.mount_refinement.end()) {
                check(false, "mount diagnostic retains arm identity");
                continue;
            }
            check(diagnostic->nominal_base_in_camera.matrix().isApprox(
                      seed.base_in_camera.matrix(), 1.0e-6F),
                  "nominal mount diagnostic equals the caller's CAD seed in primary camera coordinates");
            check(diagnostic->refined_base_in_camera.matrix().isApprox(
                      arm_link_estimate(estimate.links, seed.name, "base").in_camera.matrix(), 1.0e-6F),
                  "refined mount diagnostic uses the physical base in primary camera coordinates");
            check(diagnostic->translation_correction_m < 2.0e-5
                      && diagnostic->rotation_correction_rad < 2.0e-5,
                  "zero-information mount remains at its CAD nominal reference");
        }
        for (const auto& prior : measured) {
            check(estimate.joint_angles.at(prior.arm).at(prior.joint) == prior.angle_rad,
                  "locked joint output retains the exact encoder double including gripper");
        }
    };
    const auto first = engine.step(iteration++, feedback);
    verify(first, seeds, feedback);

    std::vector<std::vector<rbt_m3t::JointAnglePrior>> malformed { {} };
    auto missing = feedback;
    missing.pop_back();
    malformed.push_back(missing);
    auto missing_arm = feedback;
    missing_arm.erase(std::remove_if(missing_arm.begin(), missing_arm.end(),
        [](const auto& prior) { return prior.arm == "right"; }), missing_arm.end());
    malformed.push_back(missing_arm);
    auto duplicate = feedback;
    duplicate.back() = duplicate.front();
    malformed.push_back(duplicate);
    auto unknown = feedback;
    unknown.back().arm = "unknown";
    malformed.push_back(unknown);
    auto fixed = feedback;
    fixed.back().joint = "fixed_wrist";
    malformed.push_back(fixed);
    auto outside = feedback;
    outside.back().angle_rad = gripper.upper + 0.1;
    malformed.push_back(outside);
    auto nan = feedback;
    nan.back().angle_rad = std::numeric_limits<double>::quiet_NaN();
    malformed.push_back(nan);
    auto bad_sigma = feedback;
    bad_sigma.back().standard_deviation_rad = 0.0;
    malformed.push_back(bad_sigma);

    auto requested = seeds;
    std::reverse(requested.begin(), requested.end());
    for (auto& seed : requested) {
        seed.base_in_camera = Eigen::Translation3f { 0.3F, -0.1F, 0.2F }
                              * Eigen::AngleAxisf { -0.18F, Eigen::Vector3f::UnitZ() }
                              * seed.base_in_camera;
        // Reset must use fresh feedback instead of these different seed angles.
        seed.joint_angles = { { "1", 0.7 }, { "gripper", 0.5 } };
    }
    for (const auto& bad : malformed) {
        expect_invalid([&] { static_cast<void>(engine.step(iteration++, bad)); },
                       "locked step requires a complete unique finite in-range measurement on every call");
        expect_invalid([&] { engine.reinitialize(requested, iteration++, bad); },
                       "locked reset validates all feedback before changing CAD references or joint state");
        verify(engine.step(iteration++, feedback), seeds, feedback);
    }
    auto changed = feedback;
    changed[0].angle_rad = gripper.upper;
    changed[1].angle_rad = model.joints[0].upper; // Exact non-float-representable joint limit.
    changed[2].angle_rad = 0.987654321012345;
    changed[3].angle_rad = gripper.lower;
    verify(engine.step(iteration++, changed), seeds, changed);
    engine.reinitialize(requested, iteration++, changed);
    verify(engine.step(iteration++, changed), requested, changed);
    engine.reinitialize(requested, iteration++, feedback);
    verify(engine.step(iteration++, feedback), requested, feedback);
    verify(engine.step(iteration++, changed), requested, changed);
}

void encoder_reference_links_follow_feedback_and_preserve_physical_frames(bool refine_mount) {
    TempDirectory directory;
    auto model = rbt_m3t::load_robot_model(write_model(directory, model_json()));
    model.links.push_back({ "jaw", {} });
    rbt_m3t::ModelJoint jaw;
    jaw.name = "gripper";
    jaw.type = "revolute";
    jaw.parent = "wrist";
    jaw.child = "jaw";
    jaw.origin = Eigen::Translation3f { 0.04F, 0.02F, -0.01F }
                 * Eigen::AngleAxisf { 0.37F, Eigen::Vector3f::UnitX() };
    jaw.lower = -0.4;
    jaw.upper = 0.8;
    model.joints.push_back(jaw);

    rbt_m3t::ArmSeed seed;
    seed.name = "left";
    seed.fixed_base = !refine_mount;
    seed.base_in_camera = Eigen::Translation3f { 0.1F, -0.2F, 0.8F }
                          * Eigen::AngleAxisf { 0.3F, Eigen::Vector3f::UnitY() };
    seed.joint_angles = { { "1", 0.1 }, { "gripper", 0.2 } };
    rbt_m3t::EngineOptions options;
    options.cache_directory = directory.path() / "encoder_reference_cache";
    options.use_region = false;
    options.encoder_reference_links = true;
    options.mount_refinement.enabled = refine_mount;
    auto images = std::make_shared<rbt_m3t::CameraImages>();
    images->current = rectified_pair();
    rbt_m3t::TrackerEngine engine { model, { seed }, options, images };
    engine.set_up();

    const auto verify = [&](const rbt_m3t::TrackingEstimate& estimate,
                            const rbt_m3t::ArmSeed& base,
                            const std::vector<rbt_m3t::JointAnglePrior>& priors) {
        const auto expected = rbt_m3t::seeds_with_joint_priors(model, { base }, priors);
        check(!estimate.joints_locked_to_feedback, "soft reference mode leaves joints unlocked");
        check(estimate.encoder_reference_links, "estimate reports the encoder reference mode");
        check(estimate.links.size() == model.links.size(), "encoder reference links stay internal");
        check(estimate.mount_refinement.size() == (refine_mount ? 1U : 0U),
              "soft encoder mode reports mount diagnostics only when enabled");
        if (refine_mount) {
            const auto& diagnostic = estimate.mount_refinement.front();
            check(diagnostic.nominal_base_in_camera.matrix().isApprox(
                      base.base_in_camera.matrix(), 1.0e-6F)
                      && diagnostic.refined_base_in_camera.matrix().isApprox(
                          base.base_in_camera.matrix(), 2.0e-5F)
                      && diagnostic.translation_correction_m < 2.0e-5
                      && diagnostic.rotation_correction_rad < 2.0e-5,
                  "combined soft base stays at the nominal reference without visual information");
        }
        for (const auto& link : rbt_m3t::forward_kinematics(model, expected.front())) {
            check(arm_link_estimate(estimate.links, seed.name, link.link)
                      .in_camera.matrix().isApprox(link.in_camera.matrix(), 2.0e-5F),
                  "in the absence of visual information physical links follow the moving encoder reference");
        }
        for (const auto& prior : priors) {
            check(std::abs(estimate.joint_angles.at(prior.arm).at(prior.joint) - prior.angle_rad) < 2.0e-5,
                  "reported joint angle includes the reference angle");
        }
    };
    const std::vector<rbt_m3t::JointAnglePrior> first {
        { "left", "1", 0.45, 0.1 }, { "left", "gripper", -0.24, 0.05 },
    };
    const std::vector<rbt_m3t::JointAnglePrior> moved {
        { "left", "1", -0.77, 0.1 }, { "left", "gripper", jaw.upper, 0.05 },
    };
    verify(engine.step(0, first), seed, first);
    expect_invalid([&] { (void)engine.step(1, { moved.front() }); },
                   "missing feedback is rejected before reference links move");
    verify(engine.step(2, moved), seed, moved);

    auto reset = seed;
    reset.base_in_camera.translation().x() += 0.15F;
    expect_invalid([&] { engine.reinitialize({ reset }, 3, {}); },
                   "soft reference reset rejects missing feedback before changing its nominal base");
    verify(engine.step(3, moved), seed, moved);
    engine.reinitialize({ reset }, 3, first);
    verify(engine.step(4, first), reset, first);
}

void joint_priors_update_only_the_associated_seed_angles() {
    TempDirectory directory;
    const auto model = rbt_m3t::load_robot_model(write_model(directory, model_json()));
    rbt_m3t::ArmSeed left;
    left.name = "left";
    left.base_in_camera = Eigen::Translation3f { 0.2F, -0.1F, 0.4F }
                           * Eigen::AngleAxisf { 0.3F, Eigen::Vector3f::UnitY() };
    left.joint_angles = { { "1", 0.1 } };
    left.fixed_base = true;
    rbt_m3t::ArmSeed right;
    right.name = "right";
    right.base_in_camera = Eigen::Translation3f { -0.3F, 0.5F, 0.8F }
                            * Eigen::AngleAxisf { -0.2F, Eigen::Vector3f::UnitX() };
    right.joint_angles = { { "1", 0.7 } };
    const std::vector<rbt_m3t::ArmSeed> seeds { left, right };
    const std::vector<rbt_m3t::JointAnglePrior> priors {
        { "left", "1", 1.25, 0.2 },
    };

    const auto seeded = rbt_m3t::seeds_with_joint_priors(model, seeds, priors);
    check(seeded.size() == 2U, "joint-prior seed helper retains all arms");
    check(std::abs(seeded[0].joint_angles.at("1") - 1.25) < 1.0e-12,
          "joint-prior seed helper replaces the associated arm angle");
    check(std::abs(seeded[1].joint_angles.at("1") - 0.7) < 1.0e-12,
          "joint-prior seed helper preserves unassociated arm angles");
    check(seeded[0].fixed_base && !seeded[1].fixed_base,
          "joint-prior seed helper preserves each arm's fixed-base setting");
    check(seeded[0].base_in_camera.matrix().isApprox(
              left.base_in_camera.matrix(), 1.0e-6F)
              && seeded[1].base_in_camera.matrix().isApprox(
                  right.base_in_camera.matrix(), 1.0e-6F),
          "joint-prior seed helper preserves both base poses");
    check(std::abs(seeds[0].joint_angles.at("1") - 0.1) < 1.0e-12
              && std::abs(seeds[1].joint_angles.at("1") - 0.7) < 1.0e-12,
          "joint-prior seed helper leaves its input seeds unchanged");
    check(rbt_m3t::forward_kinematics(model, seeded[0]).size() == model.links.size()
              && rbt_m3t::forward_kinematics(model, seeded[1]).size() == model.links.size(),
          "joint-prior seed helper returns full-FK-valid seeds");
}

void joint_prior_seed_helper_accepts_exact_non_float_joint_limit() {
    TempDirectory directory;
    auto model = rbt_m3t::load_robot_model(write_model(directory, model_json()));
    const double upper_limit = std::acos(-1.0);
    model.joints[0].upper = upper_limit;
    check(static_cast<double>(static_cast<float>(upper_limit)) > upper_limit,
          "boundary fixture rounds above its exact double joint limit as float");

    rbt_m3t::ArmSeed left;
    left.name = "left";
    left.joint_angles = { { "1", 0.1 } };
    rbt_m3t::ArmSeed right = left;
    right.name = "right";
    const auto seeded = rbt_m3t::seeds_with_joint_priors(
        model, { left, right }, { { "left", "1", upper_limit, 0.2 } });
    check(seeded.size() == 2U
              && seeded[0].joint_angles.at("1") == upper_limit,
          "joint-prior seed helper accepts an exact URDF limit despite float rounding");
}

void joint_prior_seed_helper_rejects_invalid_mappings_and_values() {
    TempDirectory directory;
    const auto model = rbt_m3t::load_robot_model(write_model(directory, model_json()));
    rbt_m3t::ArmSeed left;
    left.name = "left";
    left.joint_angles = { { "1", 0.1 } };
    rbt_m3t::ArmSeed right = left;
    right.name = "right";
    const std::vector<rbt_m3t::ArmSeed> seeds { left, right };

    expect_invalid([&] {
        static_cast<void>(rbt_m3t::seeds_with_joint_priors(
            model, seeds, { { "left", "1", 0.2, 0.0 } }));
    }, "joint-prior seed helper rejects a zero standard deviation");
    expect_invalid([&] {
        static_cast<void>(rbt_m3t::seeds_with_joint_priors(
            model, seeds, { { "left", "1", 0.2,
                              static_cast<double>(std::numeric_limits<float>::denorm_min())
                                  / 2.0 } }));
    }, "joint-prior seed helper rejects a standard deviation that underflows as float");
    expect_invalid([&] {
        static_cast<void>(rbt_m3t::seeds_with_joint_priors(
            model, seeds, { { "left", "1", 0.2, 1.0e-30 } }));
    }, "joint-prior seed helper rejects a standard deviation with overflowing precision");
    expect_invalid([&] {
        static_cast<void>(rbt_m3t::seeds_with_joint_priors(
            model, seeds, { { "left", "1", 0.2, 1.0e30 } }));
    }, "joint-prior seed helper rejects a standard deviation with underflowing precision");
    expect_invalid([&] {
        static_cast<void>(rbt_m3t::seeds_with_joint_priors(
            model, seeds, { { "left", "1", 4.6, 0.2 } }));
    }, "joint-prior seed helper rejects an out-of-limit angle");
    expect_invalid([&] {
        static_cast<void>(rbt_m3t::seeds_with_joint_priors(
            model, seeds, { { "left", "1", std::numeric_limits<double>::quiet_NaN(), 0.2 } }));
    }, "joint-prior seed helper rejects a non-finite angle");
    expect_invalid([&] {
        static_cast<void>(rbt_m3t::seeds_with_joint_priors(
            model, seeds, { { "unknown", "1", 0.2, 0.2 } }));
    }, "joint-prior seed helper rejects an unknown arm");
    expect_invalid([&] {
        static_cast<void>(rbt_m3t::seeds_with_joint_priors(
            model, seeds,
            { { "left", "1", 0.2, 0.2 }, { "left", "1", 0.3, 0.2 } }));
    }, "joint-prior seed helper rejects duplicate arm-joint mappings");
}

void unwraps_to_nearest_valid_representative_before_clamping() {
    constexpr double lower = -1.37;
    constexpr double upper = 4.4906552;
    const auto recovered = rbt_m3t::detail::unwrap_and_clamp_joint_angle(
        -2.2831853071795862, 4.0, lower, upper);
    check(std::abs(recovered - 4.0) < 1.0e-10,
          "wrapped negative angle is unwrapped near the valid 4 radian seed");

    const auto upper_clamped = rbt_m3t::detail::unwrap_and_clamp_joint_angle(
        1.0, 4.4, lower, upper);
    check(std::abs(upper_clamped - upper) < 1.0e-10,
          "nearest representative is clamped to the upper joint limit");
    const auto lower_clamped = rbt_m3t::detail::unwrap_and_clamp_joint_angle(
        -2.0, -1.2, lower, upper);
    check(std::abs(lower_clamped - lower) < 1.0e-10,
          "wrapped result is clamped to the lower joint limit");
}

} // namespace

int main() {
    loads_json_and_resolves_relative_meshes();
    forward_kinematics_uses_origin_then_z_rotation_and_fixed_frames();
    requires_exactly_every_finite_in_limit_revolute_angle();
    rejects_invalid_axes_matrices_and_trees();
    constructor_accepts_one_or_two_arm_seeds_and_rejects_invalid_cardinality();
    tracker_keeps_fixed_bases_in_world_while_the_camera_moves();
    dual_camera_engine_validates_both_views_and_reports_primary_coordinates();
    reinitialization_validates_priors_before_mutating_poses();
    mount_refinement_validates_configuration_before_setup();
    mount_refinement_locks_complete_feedback_and_resets_nominal();
    encoder_reference_links_follow_feedback_and_preserve_physical_frames(false);
    encoder_reference_links_follow_feedback_and_preserve_physical_frames(true);
    joint_priors_update_only_the_associated_seed_angles();
    joint_prior_seed_helper_accepts_exact_non_float_joint_limit();
    joint_prior_seed_helper_rejects_invalid_mappings_and_values();
    unwraps_to_nearest_valid_representative_before_clamping();
    if (failures != 0) {
        std::cerr << failures << " model test check(s) failed\n";
        return 1;
    }
    std::cout << "All M3T model checks passed\n";
    return 0;
}
