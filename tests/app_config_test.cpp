#include <rbt_m3t/app_config.hpp>

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using rbt_m3t::LinkEstimate;
using rbt_m3t::RectifiedPair;
using rbt_m3t::TrackingEstimate;

void check(bool condition, const std::string& description) {
    if (!condition) {
        throw std::runtime_error("check failed: " + description);
    }
}

void expect_invalid(const std::function<void()>& action, const std::string& description) {
    try {
        action();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("expected invalid_argument: " + description);
}

class Fixture {
  public:
    Fixture() {
        static std::atomic<std::uint64_t> next_id { 0U };
        const auto id = next_id.fetch_add(1U, std::memory_order_relaxed);
        root_ = std::filesystem::temp_directory_path()
                / ("rbt-m3t-app-config-" + std::to_string(
                    std::chrono::steady_clock::now().time_since_epoch().count())
                   + "-" + std::to_string(id));
        std::filesystem::create_directories(root_ / "configs");
        std::filesystem::create_directories(root_ / "models");
        write(root_ / "models/robot.json", robot_json());
        write(root_ / "initialization.json", initialization_json());
        write(config_path(), config_json());
    }

    ~Fixture() {
        std::error_code ignored;
        std::filesystem::remove_all(root_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& root() const noexcept {
        return root_;
    }

    [[nodiscard]] std::filesystem::path config_path() const {
        return root_ / "configs/tracking.json";
    }

    static void write(const std::filesystem::path& path, const std::string& content) {
        std::ofstream output(path, std::ios::binary);
        check(output.good(), "open fixture file for writing");
        output << content;
        check(output.good(), "write fixture file");
    }

    [[nodiscard]] static std::string robot_json() {
        return R"json({
  "schema_version": 1,
  "name": "test_robot",
  "root_link": "base",
  "units": "metres_radians",
  "links": [
    {"name": "base", "mesh": "", "visuals": 0},
    {"name": "tip", "mesh": "", "visuals": 0}
  ],
  "joints": [
    {"name": "5", "type": "revolute", "parent": "base", "child": "tip",
     "origin": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
     "axis": [0, 0, 1], "lower": -1.37, "upper": 4.4906552}
  ]
})json";
    }

    [[nodiscard]] static std::string initialization_json(
        const std::string& left_angle = "-1.37",
        const std::string& right_angle = "4.4906552",
        const std::string& left_pose = "[1,0,0,0.25,0,1,0,0,0,0,1,0,0,0,0,1]",
        const std::string& extra_angles = "",
        bool right_fixed_base = false,
        bool include_left = true,
        bool include_right = true) {
        std::string arms;
        const auto append_arm = [&arms](std::string entry) {
            if (!arms.empty()) {
                arms += ",\n";
            }
            arms += "    " + std::move(entry);
        };
        if (include_left) {
            append_arm("{\"name\":\"so101_left\",\"base_in_camera\":" + left_pose
                       + ",\"joint_angles\":{\"5\":" + left_angle + extra_angles
                       + "},\"fixed_base\":true}");
        }
        if (include_right) {
            append_arm("{\"name\":\"so101_right\",\"base_in_camera\":[1,0,0,-0.25,0,1,0,0,0,0,1,0,0,0,0,1],"
                       "\"joint_angles\":{\"5\":" + right_angle
                       + "},\"fixed_base\":" + (right_fixed_base ? "true" : "false") + "}");
        }
        return "{\n"
               "  \"schema_version\": 1,\n"
               "  \"coordinate_frame\": \"camera/color_optical\",\n"
               "  \"serial\": \"SN-test\",\n"
               "  \"arms\": [\n"
               + arms + "\n"
               "  ]\n"
               "}\n";
    }

    [[nodiscard]] static std::string config_json(
        const std::string& model_points = "200",
        const std::string& initialization = "../initialization.json",
        const std::string& extensions = "") {
        return "{\n"
               "  \"schema_version\": 1,\n"
               "  \"robot_model\": \"../models/robot.json\",\n"
               "  \"initialization\": \"" + initialization + "\",\n"
               "  \"cache_directory\": \"../cache\",\n"
               "  \"model_points\": " + model_points + extensions + "\n"
               "}\n";
    }

    [[nodiscard]] static std::string joint_prior_mapping(
        const std::string& arm = "so101_left",
        const std::string& joint = "5",
        const std::string& source_arm = "left",
        const std::string& source_joint = "shoulder_pan",
        const std::string& scale = "0.017453292519943295",
        const std::string& offset = "0",
        const std::string& deviation = "0.1") {
        return "{\"arm\":\"" + arm + "\",\"joint\":\"" + joint
               + "\",\"source_arm\":\"" + source_arm + "\",\"source_joint\":\""
               + source_joint + "\",\"scale_rad_per_unit\":" + scale
               + ",\"offset_rad\":" + offset + ",\"standard_deviation_rad\":"
               + deviation + "}";
    }

    [[nodiscard]] static std::string joint_prior_extension(
        const std::string& mappings,
        const std::string& options = "\"service_name\":\"joint-state\","
                                         "\"controller_id\":\"controller-1\",") {
        return ",\"joint_priors\":{" + options + "\"mappings\":[" + mappings + "]}";
    }

    [[nodiscard]] static std::string camera_motion_fields(const std::string& options = "") {
        return "\"world_frame\":\"world\",\"pitch_frame\":\"camera/pitch\","
               "\"pitch_provider\":\"head-pitch\",\"depth_frame\":\"camera/depth_optical\","
               "\"mount_in_pitch\":[1,0,0,0,0,1,0,0.125,0,0,1,0,0,0,0,1],"
               "\"arm_bases\":["
               "{\"arm\":\"so101_left\",\"base_in_world\":[1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1]},"
               "{\"arm\":\"so101_right\",\"base_in_world\":[1,0,0,-1,0,1,0,0,0,0,1,0,0,0,0,1]}]"
               + options;
    }

    [[nodiscard]] static std::string camera_motion_extension(const std::string& fields) {
        return ",\"camera_motion\":{" + fields + "}";
    }

    [[nodiscard]] static std::string secondary_camera_fields(const std::string& options = "") {
        return "\"provider\":\"orbbec336\",\"serial\":\"SN-secondary\","
               "\"coordinate_frame\":\"camera2/color_optical\","
               "\"camera_in_primary\":[1,0,0,0.2,0,1,0,0,0,0,1,0,0,0,0,1]" + options;
    }

    [[nodiscard]] static std::string secondary_camera_extension(const std::string& fields) {
        return ",\"secondary_camera\":{" + fields + "}";
    }

  private:
    std::filesystem::path root_;
};

void test_load_relative_files_and_exact_joint_seeds() {
    Fixture fixture;
    const auto config = rbt_m3t::load_tracking_config(fixture.config_path());

    check(config.serial == "SN-test", "initialization serial");
    check(config.coordinate_frame == "camera/color_optical", "initialization frame");
    check(config.arms.size() == 2U, "exactly two arms");
    check(config.arms[0].name == "so101_left", "left arm name");
    check(config.arms[1].name == "so101_right", "right arm name");
    check(config.arms[0].base_in_camera.matrix()(0, 3) == 0.25F, "row-major base translation");
    check(config.arms[0].joint_angles.at("5") == -1.37, "lower custom joint 5 bound");
    check(config.arms[1].joint_angles.at("5") == 4.4906552, "upper custom joint 5 bound");
    check(config.engine.cache_directory == fixture.root() / "cache", "relative cache path");
    check(config.engine.model_points == 200, "engine default and override");
    check(config.initialization_path == fixture.root() / "initialization.json",
          "resolved initialization path");
    check(config.engine.use_region && config.engine.use_depth && !config.engine.use_texture,
          "default modality options");
    check(config.engine.measured_occlusions && !config.engine.modeled_occlusions
              && !config.engine.region_checking && config.engine.occlusion_image_size == 512,
          "default occlusion options");
    check(!config.engine.use_static_detector && !config.engine.refine_initialization,
          "default initialization flow");
    check(!config.engine.visualize_correspondences && !config.engine.visualize_optimization
              && !config.engine.visualize_results,
          "default visualization options");
    check(config.engine.constraints.empty(), "default constraints");
    check(!config.joint_prior.enabled && config.joint_prior.time_basis == "capture"
              && config.joint_prior.maximum_age_ns == 50'000'000U
              && config.joint_prior.capacity == 128U,
          "absent joint-prior configuration stays disabled with defaults");
    check(!config.camera_motion.enabled && config.camera_motion.time_basis == "capture"
              && config.camera_motion.maximum_interpolation_gap_ms == 200
              && config.camera_motion.maximum_wait_ms == 200
              && config.camera_motion.maximum_sample_age_ms == 1000
              && !config.camera_motion.allow_unknown_uncertainty,
          "absent camera-motion configuration stays disabled with defaults");
    check(!config.secondary_camera.has_value(),
          "absent secondary-camera configuration preserves single-camera mode");

    const auto initialization = rbt_m3t::load_tracking_initialization(
        config.initialization_path, config.model);
    check(initialization.arms.size() == config.arms.size()
              && initialization.coordinate_frame == config.coordinate_frame
              && initialization.serial == config.serial,
          "reload measured initialization using resolved path");
}

void test_initialization_accepts_one_arm_and_rejects_bad_cardinality() {
    Fixture fixture;
    Fixture::write(fixture.root() / "initialization.json",
                   Fixture::initialization_json("-1.37", "4.4906552",
                                                "[1,0,0,0.25,0,1,0,0,0,0,1,0,0,0,0,1]",
                                                "", false, true, false));
    const auto config = rbt_m3t::load_tracking_config(fixture.config_path());
    check(config.arms.size() == 1U && config.arms.front().name == "so101_left"
              && config.arms.front().joint_angles.at("5") == -1.37,
          "single-arm initialization loads the calibrated left seed");
    const auto initialization = rbt_m3t::load_tracking_initialization(
        config.initialization_path, config.model);
    check(initialization.arms.size() == 1U
              && initialization.arms.front().name == "so101_left",
          "single-arm initialization reload preserves the configured arm set");

    Fixture::write(fixture.root() / "initialization.json",
                   Fixture::initialization_json("-1.37", "4.4906552",
                                                "[1,0,0,0.25,0,1,0,0,0,0,1,0,0,0,0,1]",
                                                "", false, false, false));
    expect_invalid([&] { (void)rbt_m3t::load_tracking_config(fixture.config_path()); },
                   "initialization rejects an empty arm list");

    auto three_arms = Fixture::initialization_json();
    const auto closing = three_arms.rfind("  ]");
    check(closing != std::string::npos, "two-arm fixture has an arm-list terminator");
    if (closing != std::string::npos) {
        three_arms.insert(closing,
                          ",\n    {\"name\":\"so101_extra\",\"base_in_camera\":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],"
                          "\"joint_angles\":{\"5\":0},\"fixed_base\":true}\n");
        Fixture::write(fixture.root() / "initialization.json", three_arms);
        expect_invalid([&] { (void)rbt_m3t::load_tracking_config(fixture.config_path()); },
                       "initialization rejects more than two arms");
    }
}

void test_secondary_camera_config_is_optional_and_strict() {
    Fixture fixture;
    Fixture::write(fixture.config_path(), Fixture::config_json(
        "200", "../initialization.json",
        Fixture::secondary_camera_extension(Fixture::secondary_camera_fields())));
    const auto defaults = rbt_m3t::load_tracking_config(fixture.config_path());
    check(defaults.secondary_camera.has_value(), "secondary camera is parsed when configured");
    check(defaults.secondary_camera->provider == "orbbec336"
              && defaults.secondary_camera->serial == "SN-secondary"
              && defaults.secondary_camera->coordinate_frame == "camera2/color_optical",
          "secondary camera identity fields are parsed");
    check(defaults.secondary_camera->maximum_skew_ms == 20U
              && defaults.secondary_camera->time_basis == "capture",
          "secondary camera pairing defaults to bounded capture-time matching");
    check(defaults.secondary_camera->camera_in_primary.matrix()(0, 3) == 0.2F,
          "secondary camera extrinsic is parsed as row-major metres");

    Fixture::write(fixture.config_path(), Fixture::config_json(
        "200", "../initialization.json",
        Fixture::secondary_camera_extension(Fixture::secondary_camera_fields(
            ",\"maximum_skew_ms\":11,\"time_basis\":\"received\""))));
    const auto received = rbt_m3t::load_tracking_config(fixture.config_path());
    check(received.secondary_camera->maximum_skew_ms == 11U
              && received.secondary_camera->time_basis == "received",
          "secondary camera pairing options accept explicit received-time best effort");

    const auto expect_invalid_secondary = [&fixture](const std::string& fields,
                                                     const std::string& description) {
        Fixture::write(fixture.config_path(), Fixture::config_json(
            "200", "../initialization.json", Fixture::secondary_camera_extension(fields)));
        expect_invalid([&] { (void)rbt_m3t::load_tracking_config(fixture.config_path()); },
                       description);
    };
    const auto base = Fixture::secondary_camera_fields();
    expect_invalid_secondary("\"serial\":\"SN-secondary\",\"coordinate_frame\":\"camera2/color_optical\","
                             "\"camera_in_primary\":[1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1]",
                             "missing secondary provider");
    expect_invalid_secondary(base + ",\"unexpected\":true", "unknown secondary-camera field");
    expect_invalid_secondary("\"provider\":\"bad/provider\",\"serial\":\"SN-secondary\","
                             "\"coordinate_frame\":\"camera2/color_optical\","
                             "\"camera_in_primary\":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]",
                             "invalid secondary provider identifier");
    expect_invalid_secondary("\"provider\":\"orbbec336\",\"serial\":\"SN-secondary\","
                             "\"coordinate_frame\":\"camera2//color_optical\","
                             "\"camera_in_primary\":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]",
                             "invalid secondary frame identifier");
    expect_invalid_secondary("\"provider\":\"orbbec336\",\"serial\":\"SN-test\","
                             "\"coordinate_frame\":\"camera2/color_optical\","
                             "\"camera_in_primary\":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]",
                             "duplicate camera serial");
    expect_invalid_secondary("\"provider\":\"orbbec336\",\"serial\":\"SN-secondary\","
                             "\"coordinate_frame\":\"camera/color_optical\","
                             "\"camera_in_primary\":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]",
                             "duplicate camera frame");
    expect_invalid_secondary("\"provider\":\"orbbec336\",\"serial\":\"SN-secondary\","
                             "\"coordinate_frame\":\"camera2/color_optical\","
                             "\"camera_in_primary\":[2,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]",
                             "non-rigid secondary-camera extrinsic");
    expect_invalid_secondary(base + ",\"maximum_skew_ms\":0", "zero secondary skew limit");
    expect_invalid_secondary(base + ",\"maximum_skew_ms\":101", "excessive secondary skew limit");
    expect_invalid_secondary(base + ",\"maximum_skew_ms\":1.5", "non-integer secondary skew limit");
    expect_invalid_secondary(base + ",\"time_basis\":\"wall\"", "invalid secondary time basis");
}

void test_camera_motion_config_is_opt_in_and_strict() {
    Fixture fixture;
    Fixture::write(fixture.root() / "initialization.json",
                   Fixture::initialization_json("-1.37", "4.4906552",
                                                "[1,0,0,0.25,0,1,0,0,0,0,1,0,0,0,0,1]",
                                                "", true));
    Fixture::write(fixture.config_path(), Fixture::config_json(
        "200", "../initialization.json",
        Fixture::camera_motion_extension(Fixture::camera_motion_fields())));
    const auto config = rbt_m3t::load_tracking_config(fixture.config_path());
    check(config.camera_motion.enabled, "camera-motion object enables motion by default");
    check(config.camera_motion.world_frame == "world"
              && config.camera_motion.pitch_frame == "camera/pitch"
              && config.camera_motion.pitch_provider == "head-pitch"
              && config.camera_motion.depth_frame == "camera/depth_optical",
          "camera-motion frames and provider are parsed");
    check(config.camera_motion.time_basis == "capture"
              && config.camera_motion.maximum_interpolation_gap_ms == 200
              && config.camera_motion.maximum_wait_ms == 200
              && config.camera_motion.maximum_sample_age_ms == 1000
              && !config.camera_motion.allow_unknown_uncertainty,
          "camera-motion options use bounded defaults");
    check(config.camera_motion.mount_in_pitch.matrix()(1, 3) == 0.125F
              && config.camera_motion.arm_bases.size() == 2U
              && config.camera_motion.arm_bases[0].base_in_world.matrix()(0, 3) == 1.0F,
          "camera-motion transforms use row-major metres");

    Fixture::write(fixture.config_path(), Fixture::config_json(
        "200", "../initialization.json",
        Fixture::camera_motion_extension(Fixture::camera_motion_fields(
            ",\"time_basis\":\"received\",\"allow_unknown_uncertainty\":true,"
            "\"maximum_interpolation_gap_ms\":125,\"maximum_wait_ms\":800,"
            "\"maximum_sample_age_ms\":9000"))));
    const auto received = rbt_m3t::load_tracking_config(fixture.config_path());
    check(received.camera_motion.time_basis == "received"
              && received.camera_motion.allow_unknown_uncertainty
              && received.camera_motion.maximum_interpolation_gap_ms == 125
              && received.camera_motion.maximum_wait_ms == 800
              && received.camera_motion.maximum_sample_age_ms == 9000,
          "camera-motion time and uncertainty options are parsed");

    Fixture::write(fixture.config_path(), Fixture::config_json(
        "200", "../initialization.json",
        Fixture::camera_motion_extension("\"enabled\":false")));
    const auto disabled = rbt_m3t::load_tracking_config(fixture.config_path());
    check(!disabled.camera_motion.enabled, "camera-motion can be explicitly disabled");

    const auto expect_invalid_motion = [&fixture](const std::string& fields,
                                                  const std::string& description) {
        Fixture::write(fixture.config_path(), Fixture::config_json(
            "200", "../initialization.json", Fixture::camera_motion_extension(fields)));
        expect_invalid([&fixture] {
            (void)rbt_m3t::load_tracking_config(fixture.config_path());
        }, description);
    };
    const auto base = Fixture::camera_motion_fields();
    expect_invalid_motion(base + ",\"unknown\":true", "unknown camera-motion key");
    expect_invalid_motion(Fixture::camera_motion_fields(
                              ",\"world_frame\":\"other\""),
                          "duplicate camera-motion key");
    auto duplicate_frame = base;
    const auto pitch = duplicate_frame.find("camera/pitch");
    duplicate_frame.replace(pitch, std::string("camera/pitch").size(), "world");
    expect_invalid_motion(duplicate_frame, "camera-motion frames must be distinct");
    auto wrong_depth = base;
    const auto depth = wrong_depth.find("camera/depth_optical");
    wrong_depth.replace(depth, std::string("camera/depth_optical").size(),
                        "camera/color_optical");
    expect_invalid_motion(wrong_depth, "depth frame differs from initialization frame");
    auto bad_mount = base;
    const auto mount = bad_mount.find("[1,0,0,0,0,1,0,0.125");
    bad_mount.replace(mount + 1U, 1U, "2");
    expect_invalid_motion(bad_mount, "camera-motion mount is a proper rigid transform");
    auto missing_arm = base;
    const auto right = missing_arm.find(",{\"arm\":\"so101_right\"");
    missing_arm.erase(right, std::string(",{\"arm\":\"so101_right\",\"base_in_world\":[1,0,0,-1,0,1,0,0,0,0,1,0,0,0,0,1]}").size());
    expect_invalid_motion(missing_arm, "camera-motion arm bases match the arm set");
    expect_invalid_motion(Fixture::camera_motion_fields(",\"maximum_wait_ms\":1001"),
                          "camera-motion wait limit");
    expect_invalid_motion(Fixture::camera_motion_fields(",\"maximum_interpolation_gap_ms\":0"),
                          "camera-motion interpolation gap limit");
    expect_invalid_motion(Fixture::camera_motion_fields(",\"maximum_sample_age_ms\":10001"),
                          "camera-motion sample age limit");
    auto duplicate_arm = base;
    const auto right_arm = duplicate_arm.find("so101_right");
    duplicate_arm.replace(right_arm, std::string("so101_right").size(), "so101_left");
    expect_invalid_motion(duplicate_arm, "camera-motion arm names are unique");

    Fixture::write(fixture.root() / "initialization.json", Fixture::initialization_json());
    expect_invalid_motion(base, "camera-motion requires all arm seeds fixed");
}

void test_reject_missing_malformed_and_stale_config() {
    Fixture fixture;
    Fixture::write(fixture.config_path(), R"json({
      "schema_version":1,
      "robot_model":"../models/robot.json",
      "cache_directory":"../cache"
    })json");
    expect_invalid([&] { (void)rbt_m3t::load_tracking_config(fixture.config_path()); },
                   "missing initialization path");

    Fixture::write(fixture.config_path(), "{ not valid JSON");
    expect_invalid([&] { (void)rbt_m3t::load_tracking_config(fixture.config_path()); },
                   "malformed config JSON");

    Fixture::write(fixture.config_path(), Fixture::config_json("null"));
    expect_invalid([&] { (void)rbt_m3t::load_tracking_config(fixture.config_path()); },
                   "null optional field must not select a default");

    Fixture::write(fixture.config_path(), Fixture::config_json("\"200\""));
    expect_invalid([&] { (void)rbt_m3t::load_tracking_config(fixture.config_path()); },
                   "string value in integer field");
}

void test_reject_missing_init_and_invalid_physical_seeds() {
    Fixture fixture;
    Fixture::write(fixture.config_path(),
                   Fixture::config_json("200", "../missing-initialization.json"));
    expect_invalid([&] { (void)rbt_m3t::load_tracking_config(fixture.config_path()); },
                   "missing initialization file");

    Fixture::write(fixture.root() / "initialization.json",
                   Fixture::initialization_json("-1.370001"));
    Fixture::write(fixture.config_path(), Fixture::config_json());
    expect_invalid([&] { (void)rbt_m3t::load_tracking_config(fixture.config_path()); },
                   "joint angle below URDF limit");

    Fixture::write(fixture.root() / "initialization.json", Fixture::initialization_json(
        "-1.37", "4.4906552", "[-1,0,0,0.25,0,1,0,0,0,0,1,0,0,0,0,1]"));
    expect_invalid([&] { (void)rbt_m3t::load_tracking_config(fixture.config_path()); },
                   "left-handed base transform");

    Fixture::write(fixture.root() / "initialization.json", Fixture::initialization_json(
        "-1.37", "4.4906552", "[1,0,0,0.25,0,1,0,0,0,0,1,0,0,0,0,1]",
        ",\"extra_joint\":0"));
    expect_invalid([&] { (void)rbt_m3t::load_tracking_config(fixture.config_path()); },
                   "extra joint angle");
}

void test_load_nested_options_metadata_and_constraints() {
    Fixture fixture;
    const auto metadata_dir = fixture.root() / "metadata";
    std::filesystem::create_directories(metadata_dir);
    Fixture::write(metadata_dir / "region.yaml", "region: 1\n");
    Fixture::write(metadata_dir / "depth.yaml", "depth: 1\n");
    Fixture::write(metadata_dir / "texture.yaml", "texture: 1\n");
    Fixture::write(metadata_dir / "optimizer.yaml", "optimizer: 1\n");
    Fixture::write(metadata_dir / "refiner.yaml", "refiner: 1\n");
    Fixture::write(metadata_dir / "constraint.yaml", "constraint: 1\n");
    Fixture::write(fixture.config_path(), Fixture::config_json("200", "../initialization.json",
        R"json(,
  "modalities":{"region":false,"depth":true,"texture":true},
  "occlusion":{"measured":false,"modeled":true,"region_checking":true,"image_size":1024},
  "metadata":{"region":"../metadata/region.yaml","depth":"../metadata/depth.yaml",
    "texture":"../metadata/texture.yaml","optimizer":"../metadata/optimizer.yaml",
    "refiner":"../metadata/refiner.yaml"},
  "initialization_flow":{"static_detector":true,"refine":true},
  "visualization":{"correspondences":true,"optimization":true,"results":true},
  "constraints":[
    {"name":"left_hard","arm":"so101_left","link1":"base","link2":"tip",
     "type":"hard","metadata":"../metadata/constraint.yaml"},
    {"name":"right_soft","arm":"so101_right","link1":"tip","link2":"base",
     "type":"soft","metadata":"../metadata/constraint.yaml"}
  ])json"));

    const auto config = rbt_m3t::load_tracking_config(fixture.config_path());
    check(!config.engine.use_region && config.engine.use_depth && config.engine.use_texture,
          "configured modality options");
    check(!config.engine.measured_occlusions && config.engine.modeled_occlusions
              && config.engine.region_checking && config.engine.occlusion_image_size == 1024,
          "configured occlusion options");
    check(config.engine.region_metadata == metadata_dir / "region.yaml"
              && config.engine.depth_metadata == metadata_dir / "depth.yaml"
              && config.engine.texture_metadata == metadata_dir / "texture.yaml"
              && config.engine.optimizer_metadata == metadata_dir / "optimizer.yaml"
              && config.engine.refiner_metadata == metadata_dir / "refiner.yaml",
          "resolved metadata paths");
    check(config.engine.use_static_detector && config.engine.refine_initialization,
          "configured initialization flow");
    check(config.engine.visualize_correspondences && config.engine.visualize_optimization
              && config.engine.visualize_results,
          "configured visualization options");
    check(config.engine.constraints.size() == 2U, "loaded constraints");
    check(config.engine.constraints[0].name == "left_hard"
              && config.engine.constraints[0].arm == "so101_left"
              && config.engine.constraints[0].link1 == "base"
              && config.engine.constraints[0].link2 == "tip"
              && !config.engine.constraints[0].soft,
          "hard constraint mapping");
    check(config.engine.constraints[1].name == "right_soft"
              && config.engine.constraints[1].arm == "so101_right"
              && config.engine.constraints[1].soft,
          "soft constraint mapping");
    check(config.engine.constraints[0].metadata == metadata_dir / "constraint.yaml",
          "resolved constraint metadata path");
}

void test_reject_invalid_nested_options_constraints_and_metadata_paths() {
    Fixture fixture;
    Fixture::write(fixture.root() / "constraint.yaml", "constraint: 1\n");
    const auto expect_config_invalid = [&fixture](const std::string& extensions,
                                                  const std::string& description) {
        Fixture::write(fixture.config_path(), Fixture::config_json("200", "../initialization.json",
                                                                    extensions));
        expect_invalid([&fixture] {
            (void)rbt_m3t::load_tracking_config(fixture.config_path());
        }, description);
    };

    expect_config_invalid(R"json(,"modalities":{"region":"true"})json",
                          "modality must be a boolean");
    expect_config_invalid(R"json(,"modalities":{"unknown":true})json",
                          "unknown modality option");
    expect_config_invalid(R"json(,"modalities":{"region":false,"depth":false,"texture":false})json",
                          "all modalities disabled");
    expect_config_invalid(R"json(,"occlusion":{"image_size":63})json",
                          "occlusion image size lower bound");
    expect_config_invalid(R"json(,"initialization_flow":{"refine":1.0})json",
                          "initialization flow boolean type");
    expect_config_invalid(R"json(,"visualization":{"unknown":true})json",
                          "unknown visualization option");
    expect_config_invalid(R"json(,"constraints":{})json", "constraints must be an array");
    expect_config_invalid(R"json(,"constraints":[{
      "name":"c", "arm":"missing", "link1":"base",
      "link2":"tip", "type":"hard", "metadata":"../constraint.yaml"
    }])json", "unknown constraint arm");
    expect_config_invalid(R"json(,"constraints":[{
      "name":"c", "arm":"so101_left", "link1":"base",
      "link2":"missing", "type":"hard", "metadata":"../constraint.yaml"
    }])json", "unknown constraint link");
    expect_config_invalid(R"json(,"constraints":[{
      "name":"c", "arm":"so101_left", "link1":"base",
      "link2":"base", "type":"hard", "metadata":"../constraint.yaml"
    }])json", "same constraint link");
    expect_config_invalid(R"json(,"constraints":[{
      "name":"c", "arm":"so101_left", "link1":"base",
      "link2":"tip", "type":"other", "metadata":"../constraint.yaml"
    }])json", "unknown constraint type");
    expect_config_invalid(R"json(,"constraints":[
      {"name":"same", "arm":"so101_left", "link1":"base", "link2":"tip",
       "type":"hard", "metadata":"../constraint.yaml"},
      {"name":"same", "arm":"so101_right", "link1":"base", "link2":"tip",
       "type":"soft", "metadata":"../constraint.yaml"}
    ])json", "duplicate constraint name");
    expect_config_invalid(R"json(,"constraints":[{
      "name":"c", "arm":"so101_left", "link1":"base",
      "link2":"tip", "type":"hard"
    }])json", "missing required constraint metadata");

    expect_config_invalid(R"json(,"metadata":{"region":"../missing.yaml"})json",
                          "missing metadata file");
    std::filesystem::create_directories(fixture.root() / "metadata-directory");
    expect_config_invalid(R"json(,"metadata":{"region":"../metadata-directory"})json",
                          "metadata must be a regular file");
    Fixture::write(fixture.root() / "unreadable.yaml", "ignored: true\n");
    std::filesystem::permissions(fixture.root() / "unreadable.yaml",
                                 std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::replace);
    expect_config_invalid(R"json(,"metadata":{"region":"../unreadable.yaml"})json",
                          "metadata path must be readable");

    const auto absolute_metadata = (fixture.root() / "constraint.yaml").string();
    expect_config_invalid(",\"metadata\":{\"region\":\"" + absolute_metadata + "\"}",
                          "metadata path must be relative");

    std::string too_many_constraints = R"json(,"constraints":[)json";
    for (int index = 0; index < 65; ++index) {
        if (index != 0) {
            too_many_constraints += ',';
        }
        too_many_constraints += "{}";
    }
    too_many_constraints += "]";
    expect_config_invalid(too_many_constraints, "constraint count limit");
}

void test_joint_prior_config_is_explicit_and_bounded() {
    Fixture fixture;
    const auto mapping = Fixture::joint_prior_mapping();
    Fixture::write(fixture.config_path(), Fixture::config_json(
        "200", "../initialization.json",
        Fixture::joint_prior_extension(mapping,
            "\"service_name\":\"joint-state\",\"controller_id\":\"controller-1\","
            "\"required\":false,\"time_basis\":\"received\","
            "\"maximum_age_ms\":75,\"capacity\":16,")));
    const auto config = rbt_m3t::load_tracking_config(fixture.config_path());
    check(config.joint_prior.enabled, "joint prior object enables priors by default");
    check(!config.joint_prior.required, "joint prior required defaults false");
    check(config.joint_prior.initialize_from_measurement,
          "joint prior measurement initialization defaults true");
    check(config.joint_prior.service_name == "joint-state"
              && config.joint_prior.controller_id == "controller-1",
          "joint prior service and controller identifiers");
    check(config.joint_prior.time_basis == "received", "explicit receipt time basis");
    check(config.joint_prior.maximum_age_ns == 75'000'000U
              && config.joint_prior.capacity == 16U,
          "joint prior age conversion and capacity");
    check(config.joint_prior.mappings.size() == 1U
              && config.joint_prior.mappings[0].joint == "5",
          "partial joint mapping is retained without inferred wrist mappings");

    Fixture::write(fixture.config_path(), Fixture::config_json(
        "200", "../initialization.json", Fixture::joint_prior_extension(mapping)));
    const auto defaults = rbt_m3t::load_tracking_config(fixture.config_path());
    check(defaults.joint_prior.time_basis == "capture"
              && defaults.joint_prior.maximum_age_ns == 50'000'000U
              && defaults.joint_prior.capacity == 128U,
          "enabled joint-prior defaults are strict capture, 50 ms, and 128 samples");

    Fixture::write(fixture.config_path(), Fixture::config_json(
        "200", "../initialization.json", R"json(,"joint_priors":{"enabled":false})json"));
    const auto disabled = rbt_m3t::load_tracking_config(fixture.config_path());
    check(!disabled.joint_prior.enabled, "explicit disabled joint-prior object");
    check(disabled.joint_prior.mappings.empty(), "disabled priors do not invent mappings");

    const auto expect_invalid_joint_prior = [&fixture](const std::string& joint_prior,
                                                        const std::string& description) {
        Fixture::write(fixture.config_path(), Fixture::config_json(
            "200", "../initialization.json", ",\"joint_priors\":" + joint_prior));
        expect_invalid([&fixture] {
            (void)rbt_m3t::load_tracking_config(fixture.config_path());
        }, description);
    };
    expect_invalid_joint_prior(R"json({"service_name":"s","controller_id":"c",)json"
                               R"json("mappings":[{"arm":"so101_left","joint":"5",)json"
                               R"json("source_arm":"left","source_joint":"not_a_joint",)json"
                               R"json("scale_rad_per_unit":1,"offset_rad":0,)json"
                               R"json("standard_deviation_rad":0.1}]})json",
                              "unknown canonical source joint");
    expect_invalid_joint_prior(R"json({"service_name":"s","controller_id":"c",)json"
                               R"json("mappings":[)json" + Fixture::joint_prior_mapping()
                               + "," + Fixture::joint_prior_mapping() + "]}",
                              "duplicate target arm and joint mapping");
    expect_invalid_joint_prior(R"json({"service_name":"s","controller_id":"c",)json"
                               R"json("mappings":[)json"
                               + Fixture::joint_prior_mapping("so101_left", "5", "left",
                                                              "shoulder_pan", "1", "0", "-0.1")
                               + "]}",
                              "negative mapping standard deviation");
    expect_invalid_joint_prior(R"json({"unknown":true})json", "unknown joint-prior key");
    expect_invalid_joint_prior(R"json({"service_name":"s","controller_id":"c",)json"
                               R"json("maximum_age_ms":1001,"mappings":[)json"
                               + Fixture::joint_prior_mapping() + "]}",
                              "joint-prior age outside range");
    expect_invalid_joint_prior(R"json({"service_name":"s","controller_id":"c",)json"
                               R"json("capacity":513,"mappings":[)json"
                               + Fixture::joint_prior_mapping() + "]}",
                              "joint-prior capacity outside range");
    expect_invalid_joint_prior(R"json({"service_name":"s","controller_id":"c",)json"
                               R"json("mappings":[{"arm":"so101_left","joint":"5",)json"
                               R"json("source_arm":"left","source_joint":"shoulder_pan",)json"
                               R"json("scale_rad_per_unit":0,"offset_rad":0,)json"
                               R"json("standard_deviation_rad":0.1}]})json",
                              "zero mapping scale");
    expect_invalid_joint_prior(R"json({"service_name":"s","controller_id":"c",)json"
                               R"json("mappings":[)json"
                               + Fixture::joint_prior_mapping("so101_left", "missing") + "]}",
                              "unknown model joint");
}

[[nodiscard]] rbt::time::ClockIdentity device_clock() {
    std::array<std::uint8_t, 48> source {};
    source[0] = 1U;
    auto clock = rbt::time::ClockIdentity::create(
        rbt::time::ClockKind::Device, rbt::time::ClockSourceId(std::move(source)),
        std::numeric_limits<std::uint64_t>::max());
    check(clock.has_value(), "valid device clock identity");
    return std::move(clock).value();
}

[[nodiscard]] RectifiedPair output_pair(bool known_uncertainty = true) {
    const auto clock = device_clock();
    constexpr std::int64_t color_ns = 9007199254740976LL;
    constexpr std::int64_t depth_ns = 9007199254740993LL;
    RectifiedPair pair {};
    pair.timing.capture = rbt::time::Timestamp { clock, depth_ns };
    pair.timing.received = rbt::time::Timestamp { clock, depth_ns + 100 };
    pair.timing.capture_uncertainty_ns = known_uncertainty
        ? std::optional<std::uint64_t> { 18446744073709551614ULL }
        : std::nullopt;
    pair.color_capture = rbt::time::Timestamp { clock, color_ns };
    pair.skew_ns = 17U;
    pair.frame_id = "camera/color_optical";
    return pair;
}

void test_visual_link_weights_config_is_optional_and_bounded() {
    Fixture fixture;
    check(rbt_m3t::load_tracking_config(fixture.config_path()).engine.visual_link_weights.empty(),
          "absent visual link weights preserve the unweighted configuration");
    auto robot = Fixture::robot_json();
    std::size_t position = 0U;
    while ((position = robot.find("\"tip\"", position)) != std::string::npos) {
        robot.replace(position, 5U, "\"gripper\"");
        position += 9U;
    }
    const std::string old_mesh = "\"name\": \"gripper\", \"mesh\": \"\", \"visuals\": 0";
    position = robot.find(old_mesh);
    check(position != std::string::npos, "fixture has a gripper link");
    robot.replace(position, old_mesh.size(),
                  "\"name\": \"gripper\", \"mesh\": \"gripper.obj\", \"visuals\": 1");
    Fixture::write(fixture.root() / "models/robot.json", robot);
    Fixture::write(fixture.root() / "models/gripper.obj",
                   "v 0 0 0\nv 0.01 0 0\nv 0 0.01 0\nf 1 2 3\n");
    const auto load = [&fixture](const std::string& weights) {
        Fixture::write(fixture.config_path(), Fixture::config_json(
            "200", "../initialization.json", ",\"visual_link_weights\":" + weights));
        return rbt_m3t::load_tracking_config(fixture.config_path());
    };
    const auto config = load("{\"gripper\":8}");
    check(config.engine.visual_link_weights.size() == 1U
              && config.engine.visual_link_weights.at("gripper") == 8.0,
          "gripper visual residuals can receive an explicit priority multiplier");
    check(!config.engine.mount_refinement.enabled
              && config.engine.mount_refinement.translation_standard_deviation_m == 0.02
              && config.engine.mount_refinement.rotation_standard_deviation_rad
                     == 0.08726646259971647,
          "visual residual weights leave CAD mount-prior options unchanged");
    for (const auto* weight : { "0.01", "100" }) {
        const auto boundary = load("{\"gripper\":" + std::string(weight) + "}");
        check(boundary.engine.visual_link_weights.at("gripper") == std::stod(weight),
              "visual weight range includes both endpoints");
    }
    check(load("{}").engine.visual_link_weights.empty(), "an empty weight object is valid");
    for (const auto* value : { "[]", "[1]", "1", "\"gripper\"",
                               "{\"\":1}", "{\"bad name\":1}", "{\"arm/gripper\":1}",
                               "{\"unknown\":1}", "{\"base\":1}",
                               "{\"gripper\":1,\"gripper\":2}" }) {
        expect_invalid([&] { (void)load(value); },
                       "visual weights reject invalid objects, duplicate names and nonvisual links");
    }
    for (const auto* weight : { "0", "-1", "0.0099", "100.001", "1e999", "\"8\"", "[]", "{}" }) {
        expect_invalid([&] { (void)load("{\"gripper\":" + std::string(weight) + "}"); },
                       "visual weights must be finite numeric multipliers within the range");
    }
}

void test_mount_refinement_config_requires_complete_feedback_and_static_camera() {
    Fixture fixture;
    check(!rbt_m3t::load_tracking_config(fixture.config_path()).engine.mount_refinement.enabled,
          "mount refinement is disabled by default");
    auto initialization = Fixture::initialization_json();
    const auto fixed = initialization.find("\"fixed_base\":true");
    check(fixed != std::string::npos, "fixture has a fixed left base");
    initialization.replace(fixed, std::string("\"fixed_base\":true").size(),
                           "\"fixed_base\":false");
    Fixture::write(fixture.root() / "initialization.json", initialization);
    const auto mappings = Fixture::joint_prior_mapping() + ","
                          + Fixture::joint_prior_mapping("so101_right", "5", "right");
    const auto priors = [&mappings](const std::string& extra = "") {
        return Fixture::joint_prior_extension(mappings,
            "\"service_name\":\"joint-state\",\"controller_id\":\"controller-1\","
            "\"required\":true," + extra);
    };
    const auto refinement = [](const std::string& fields = "\"enabled\":true") {
        return ",\"mount_refinement\":{" + fields + "}";
    };
    const auto load = [&fixture](const std::string& extensions) {
        Fixture::write(fixture.config_path(), Fixture::config_json(
            "200", "../initialization.json", extensions));
        return rbt_m3t::load_tracking_config(fixture.config_path());
    };
    const auto config = load(priors() + refinement());
    check(config.engine.mount_refinement.enabled
              && config.engine.mount_refinement.translation_standard_deviation_m == 0.02
              && config.engine.mount_refinement.rotation_standard_deviation_rad
                     == 0.08726646259971647
              && config.joint_prior.mappings.size() == 2U,
          "mount refinement loads complete feedback mappings and default sigmas");
    const std::string encoder_references = ",\"encoder_reference_links\":true";
    const auto combined = load(encoder_references + priors() + refinement());
    check(combined.engine.mount_refinement.enabled && combined.engine.encoder_reference_links
              && std::all_of(combined.arms.begin(), combined.arms.end(),
                             [](const auto& arm) { return !arm.fixed_base; }),
          "mount refinement accepts free bases together with soft encoder reference links");
    expect_invalid([&] { (void)load(encoder_references + refinement()); },
                   "combined mount and encoder references require joint feedback");
    expect_invalid([&] {
        (void)load(encoder_references + priors("\"initialize_from_measurement\":false,")
                   + refinement());
    }, "combined mount and encoder references require measured initialization");
    expect_invalid([&] {
        (void)load(encoder_references + priors() + refinement()
                   + Fixture::camera_motion_extension(Fixture::camera_motion_fields()));
    }, "combined mount and encoder references require a static camera");
    for (const auto& values : { std::pair { "0.0001", "0.0001" },
                                std::pair { "0.2", "0.785398" } }) {
        const auto bounds = load(priors() + refinement(
            "\"enabled\":true,\"translation_standard_deviation_m\":"
            + std::string(values.first) + ",\"rotation_standard_deviation_rad\":"
            + values.second));
        check(bounds.engine.mount_refinement.translation_standard_deviation_m > 0.0,
              "mount refinement sigma bounds are inclusive");
    }
    const auto reject = [&load](const std::string& extensions, const std::string& description) {
        expect_invalid([&] { (void)load(extensions); }, description);
    };
    reject(refinement(), "refinement needs joint priors");
    reject(Fixture::joint_prior_extension(mappings) + refinement(),
           "refinement needs required joint priors");
    reject(priors("\"enabled\":false,") + refinement(), "refinement needs enabled priors");
    reject(priors("\"initialize_from_measurement\":false,") + refinement(),
           "refinement must initialize from measured feedback");
    reject(Fixture::joint_prior_extension(Fixture::joint_prior_mapping(),
               "\"service_name\":\"joint-state\",\"controller_id\":\"controller-1\","
               "\"required\":true,") + refinement(),
           "refinement rejects an unmapped arm");
    reject(priors() + refinement() + Fixture::camera_motion_extension(Fixture::camera_motion_fields()),
           "refinement rejects a dynamic camera");
    for (const auto* flow : { "static_detector", "refine" }) {
        reject(priors() + refinement() + ",\"initialization_flow\":{\"" + flow + "\":true}",
               "refinement rejects detector and initialization refinement");
    }
    Fixture::write(fixture.root() / "initialization.json", Fixture::initialization_json());
    reject(priors() + refinement(), "refinement rejects a fixed left base");
    reject(encoder_references + priors() + refinement(),
           "combined mount and encoder references reject a fixed left base");
    auto right_fixed_initialization = initialization;
    const auto right_fixed = right_fixed_initialization.rfind("\"fixed_base\":false");
    right_fixed_initialization.replace(right_fixed, std::string("\"fixed_base\":false").size(),
                                        "\"fixed_base\":true");
    Fixture::write(fixture.root() / "initialization.json", right_fixed_initialization);
    reject(priors() + refinement(), "refinement rejects a fixed right base");
    reject(encoder_references + priors() + refinement(),
           "combined mount and encoder references reject a fixed right base");
    Fixture::write(fixture.root() / "initialization.json", initialization);
    for (const auto* key : { "translation_standard_deviation_m", "rotation_standard_deviation_rad" }) {
        for (const auto* value : { "0", "-0.01", "0.00009", "1", "1e999", "\"0.02\"" }) {
            reject(priors() + refinement("\"enabled\":true,\"" + std::string(key) + "\":" + value),
                   "refinement rejects nonnumeric, nonfinite and out-of-range sigmas");
        }
    }
    reject(refinement("\"enabled\":false,\"translation_standard_deviation_m\":0"),
           "explicit disabled refinement still validates sigma fields");
    reject(refinement("\"enabled\":\"true\""), "refinement enabled must be boolean");
    reject(refinement("\"enabled\":false,\"typo\":1"), "refinement rejects unknown keys");
    reject(",\"mount_refinement\":[]", "refinement must be an object");
    reject(refinement("\"enabled\":false,\"enabled\":true"), "refinement rejects duplicate keys");
    Fixture::write(fixture.root() / "initialization.json",
                   Fixture::initialization_json("-1.37", "4.4906552",
                       "[1,0,0,0.25,0,1,0,0,0,0,1,0,0,0,0,1]", "", true));
    const auto disabled = load(refinement("\"enabled\":false")
                               + Fixture::camera_motion_extension(Fixture::camera_motion_fields()));
    check(!disabled.engine.mount_refinement.enabled && disabled.arms.front().fixed_base
              && disabled.camera_motion.enabled && !disabled.joint_prior.enabled,
          "disabled refinement preserves fixed bases, dynamic cameras and absent priors");

    // A second revolute joint proves coverage is per joint as well as per arm.
    auto robot = Fixture::robot_json();
    const auto links_end = robot.find("\n  ],");
    robot.insert(links_end, ",\n    {\"name\":\"tip2\",\"mesh\":\"\",\"visuals\":0}");
    const auto joints_end = robot.rfind("\n  ]");
    robot.insert(joints_end, R"json(,
    {"name":"6","type":"revolute","parent":"tip","child":"tip2",
     "origin":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
     "axis":[0,0,1],"lower":-1,"upper":1})json");
    Fixture::write(fixture.root() / "models/robot.json", robot);
    auto two_joint_initialization = initialization;
    std::size_t position = 0U;
    for (int arm = 0; arm < 2; ++arm) {
        position = two_joint_initialization.find("},\"fixed_base\"", position);
        two_joint_initialization.insert(position, ",\"6\":0");
        position += std::string(",\"6\":0}").size();
    }
    Fixture::write(fixture.root() / "initialization.json", two_joint_initialization);
    reject(priors() + refinement(), "refinement rejects an unmapped revolute joint on each arm");
    reject(encoder_references + priors() + refinement(),
           "combined mount and encoder references reject an unmapped revolute joint");
    const auto complete_mappings = mappings + ","
        + Fixture::joint_prior_mapping("so101_left", "6", "left", "elbow_flex") + ","
        + Fixture::joint_prior_mapping("so101_right", "6", "right", "elbow_flex");
    const auto complete = load(Fixture::joint_prior_extension(complete_mappings,
        "\"service_name\":\"joint-state\",\"controller_id\":\"controller-1\","
        "\"required\":true,") + refinement());
    check(complete.joint_prior.mappings.size() == 4U,
          "refinement accepts all revolute joints mapped on every arm");
}

void test_serializer_preserves_integer_timing_and_row_major_pose() {
    TrackingEstimate estimate;
    Eigen::Affine3f pose = Eigen::Affine3f::Identity();
    pose.matrix()(0, 0) = 0.0F;
    pose.matrix()(0, 1) = -1.0F;
    pose.matrix()(1, 0) = 1.0F;
    pose.matrix()(1, 1) = 0.0F;
    pose.matrix()(0, 3) = 2.0F;
    pose.matrix()(1, 3) = 3.0F;
    pose.matrix()(2, 3) = 4.0F;
    estimate.links.push_back(LinkEstimate { "so101_left", "tip", pose });
    estimate.joint_angles["so101_left"]["5"] = -1.37;

    const auto json = rbt_m3t::tracking_json(estimate, output_pair(), 9007199254740995ULL, 7U);
    check(json.find("\"capture_ns\":9007199254740993") != std::string::npos,
          "exact depth capture timestamp");
    check(json.find("\"color_capture_ns\":9007199254740976") != std::string::npos,
          "exact color capture timestamp");
    check(json.find("\"frame_sequence\":9007199254740995") != std::string::npos,
          "exact frame sequence");
    check(json.find("\"capture_clock_generation\":18446744073709551615")
              != std::string::npos,
          "exact clock generation");
    check(json.find("\"uncertainty_ns\":18446744073709551614") != std::string::npos,
          "exact capture uncertainty");
    check(json.find("\"frame_id\":\"so101_left/tip\",\"pose\":[0,-1,0,2,1,0,0,3,0,0,1,4,0,0,0,1]")
              != std::string::npos,
          "row-major link pose");
    check(json.find("\"algorithm\":\"m3t_region_depth_articulated\"") != std::string::npos,
          "legacy algorithm for region and depth");
    check(json.find("\"modalities\":[\"region\",\"depth\"]") != std::string::npos,
          "default modality names in output");
    check(json.find("\"joint_priors\"") == std::string::npos,
          "disabled joint priors are omitted from output");

    auto host_pair = output_pair();
    auto host = rbt::time::HostClock::create();
    check(static_cast<bool>(host), "host clock available");
    host_pair.timing.capture->clock = host.value().monotonic_identity();
    host_pair.color_capture.clock = host.value().monotonic_identity();
    host_pair.timing.received.clock = host.value().monotonic_identity();
    const auto host_json = rbt_m3t::tracking_json(estimate, host_pair, 1, 0);
    check(host_json.find("\"capture_clock_kind\":\"HostMonotonic\"") != std::string::npos,
          "tracking output preserves the unified camera host clock");

    const auto pair = output_pair();
    rbt_m3t::CameraMotionSelection motion;
    motion.camera_in_world = Eigen::Affine3f::Identity();
    motion.world_frame = "world";
    motion.time_basis = "capture";
    motion.lookup.requested_timestamp = *pair.timing.capture;
    motion.lookup.timing_uncertainty_ns = 17U;
    motion.pitch_sample_timestamp = rbt::time::Timestamp {
        pair.timing.capture->clock, pair.timing.capture->ns - 10 };
    motion.pitch_sample_offset_ns = -10;
    motion.pitch_sample_offset_abs_ns = 10U;
    const auto with_motion = rbt_m3t::tracking_json(estimate, pair, 1U, 0U, &motion);
    check(with_motion.find("\"camera_motion\":{\"enabled\":true,\"status\":\"used\",\"used\":true")
              != std::string::npos,
          "camera-motion usage status is serialized");
    check(with_motion.find("\"selection_policy\":\"nearest\"") != std::string::npos
              && with_motion.find("\"pitch_sample_offset_ns\":-10,\"pitch_sample_offset_abs_ns\":10")
                     != std::string::npos,
          "nearest pitch-sample timing is serialized");
    check(with_motion.find("\"camera_in_world\":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]")
              != std::string::npos,
          "camera pose is serialized in row-major form");

    auto secondary = output_pair();
    secondary.frame_id = "camera2/color_optical";
    secondary.timing.capture->ns += 10;
    secondary.timing.received.ns += 10;
    secondary.color_capture.ns += 10;
    rbt_m3t::SecondaryCameraConfig secondary_config;
    secondary_config.provider = "orbbec336";
    secondary_config.serial = "SN-secondary";
    secondary_config.coordinate_frame = secondary.frame_id;
    secondary_config.camera_in_primary.matrix()(0, 3) = 0.2F;
    const auto dual_json = rbt_m3t::tracking_json(
        estimate, pair, 1U, 0U, nullptr, &secondary, &secondary_config);
    check(dual_json.find("\"secondary_camera\":{\"provider\":\"orbbec336\","
                         "\"serial\":\"SN-secondary\",\"frame_id\":\"camera2/color_optical\"")
              != std::string::npos,
          "dual output identifies the configured secondary camera");
    check(dual_json.find("\"time_basis\":\"capture\",\"maximum_skew_ns\":20000000,"
                         "\"cross_view_skew_ns\":10") != std::string::npos,
          "dual output reports cross-view capture basis and skew");
    check(dual_json.find("\"camera_in_primary\":[1,0,0,0.20000000298023224,0,1,0,0,0,0,1,0,0,0,0,1]")
              != std::string::npos,
          "dual output reports the row-major fixed rig extrinsic");
    check(dual_json.find("\"primary_timing\":{\"capture\":{\"timestamp_ns\":9007199254740993,"
                         "\"clock_kind\":\"Device\",\"clock_source\":")
              != std::string::npos
              && dual_json.find("\"secondary_timing\":{\"capture\":{\"timestamp_ns\":9007199254741003,"
                               "\"clock_kind\":\"Device\",\"clock_source\":")
                     != std::string::npos
              && dual_json.find("\"received\":{\"timestamp_ns\":9007199254741093,"
                                "\"clock_kind\":\"Device\",\"clock_source\":")
                     != std::string::npos,
          "dual output reports full capture and received clock identity for both views");

    secondary_config.time_basis = "received";
    auto received_clock_result = rbt::time::HostClock::create();
    check(static_cast<bool>(received_clock_result), "host clock available for received pairing");
    const auto received_clock = received_clock_result.value().monotonic_identity();
    auto received_primary = pair;
    auto received_secondary = secondary;
    received_primary.timing.received.clock = received_clock;
    received_secondary.timing.received.clock = received_clock;
    const auto received_dual_json = rbt_m3t::tracking_json(
        estimate, received_primary, 1U, 0U, nullptr, &received_secondary, &secondary_config);
    check(received_dual_json.find("\"time_basis\":\"received\",\"maximum_skew_ns\":20000000,"
                                  "\"cross_view_skew_ns\":10") != std::string::npos,
          "received-time pairing records shared host clock identity");
    expect_invalid([&] {
        (void)rbt_m3t::tracking_json(estimate, pair, 1U, 0U,
                                     nullptr, &secondary, &secondary_config);
    }, "received-time output requires HostMonotonic timestamps");
    secondary_config.maximum_skew_ms = 1U;
    received_secondary.timing.received.ns += 2'000'000;
    expect_invalid([&] {
        (void)rbt_m3t::tracking_json(estimate, received_primary, 1U, 0U,
                                     nullptr, &received_secondary, &secondary_config);
    }, "dual output rejects cross-view skew beyond the configured basis limit");
    secondary_config.maximum_skew_ms = 0U;
    expect_invalid([&] {
        (void)rbt_m3t::tracking_json(estimate, received_primary, 1U, 0U,
                                     nullptr, &received_secondary, &secondary_config);
    }, "dual output rejects an unbounded or zero skew limit");

    const auto unknown = rbt_m3t::tracking_json(estimate, output_pair(false), 1U, 0U);
    check(unknown.find("\"uncertainty_ns\":null") != std::string::npos,
          "unknown uncertainty remains null");

    estimate.modalities = { "texture" };
    const auto texture = rbt_m3t::tracking_json(estimate, output_pair(), 1U, 0U);
    check(texture.find("\"algorithm\":\"m3t_articulated\"") != std::string::npos,
          "general algorithm for other modality sets");
    check(texture.find("\"modalities\":[\"texture\"]") != std::string::npos,
          "configured modality names in output");

    estimate.joint_prior.enabled = true;
    estimate.joint_prior.status = "applied";
    estimate.joint_prior.time_basis = "received";
    estimate.joint_prior.controller_id = "controller\"one";
    estimate.joint_prior.driver_session = "session-1";
    estimate.joint_prior.sequence = 42U;
    estimate.joint_prior.age_ns = 7U;
    estimate.joint_prior.capture_time_associated = false;
    estimate.joint_prior.priors.push_back(
        rbt_m3t::JointAnglePrior { "so101_left", "5", 0.25, 0.1 });
    estimate.joint_prior.rejected_mappings.push_back("wrist_roll: unavailable");
    const auto with_prior = rbt_m3t::tracking_json(estimate, output_pair(), 1U, 0U);
    check(with_prior.find("\"joint_priors\":{\"enabled\":true,\"status\":\"applied\"")
              != std::string::npos,
          "joint prior status is serialized");
    check(with_prior.find("\"time_basis\":\"received\",\"controller_id\":\"controller\\\"one\"")
              != std::string::npos,
          "receipt time basis and escaped controller id are serialized");
    check(with_prior.find("\"driver_session\":\"session-1\",\"sequence\":42,\"age_ns\":7")
              != std::string::npos,
          "joint prior source session, sequence, and age are serialized");
    check(with_prior.find("\"capture_time_associated\":false,\"prior_count\":1")
              != std::string::npos,
          "receipt association flag and count are serialized");
    check(with_prior.find("\"arm\":\"so101_left\",\"joint\":\"5\",\"angle_rad\":0.25,\"standard_deviation_rad\":0.10000000000000001")
              != std::string::npos,
          "joint prior angle and deviation are serialized");
    check(with_prior.find("\"rejected_mappings\":[\"wrist_roll: unavailable\"]")
              != std::string::npos,
          "mapping rejection reasons are serialized");
    estimate.joint_prior.age_ns.reset();
    const auto unknown_age = rbt_m3t::tracking_json(estimate, output_pair(), 1U, 0U);
    check(unknown_age.find("\"age_ns\":null") != std::string::npos,
          "unknown joint prior age remains null");

    estimate.modalities.clear();
    expect_invalid([&] { (void)rbt_m3t::tracking_json(estimate, output_pair(), 0U, 0U); },
                   "empty output modalities");
    estimate.modalities = { "unknown" };
    expect_invalid([&] { (void)rbt_m3t::tracking_json(estimate, output_pair(), 0U, 0U); },
                   "unknown output modality");
    estimate.modalities = { "depth", "depth" };
    expect_invalid([&] { (void)rbt_m3t::tracking_json(estimate, output_pair(), 0U, 0U); },
                   "duplicate output modality");
}

void test_mount_refinement_output_is_strict_and_preserves_all_joint_priors() {
    TrackingEstimate estimate;
    estimate.links.push_back(LinkEstimate { "so101_left", "tip", Eigen::Affine3f::Identity() });
    check(rbt_m3t::tracking_json(estimate, output_pair(), 1U, 0U).find("mount_refinement")
              == std::string::npos,
          "legacy output omits mount refinement entirely");
    estimate.links.push_back(LinkEstimate { "so101_right", "tip", Eigen::Affine3f::Identity() });
    estimate.joints_locked_to_feedback = true;
    rbt_m3t::MountRefinementEstimate left;
    left.arm = "so101_left";
    left.nominal_base_in_camera.translation().x() = 0.25F;
    left.refined_base_in_camera.translation().x() = 0.27F;
    left.refined_base_in_camera.linear() =
        Eigen::AngleAxisf(0.03F, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    left.translation_correction_m = 0.02;
    left.rotation_correction_rad = 0.03;
    auto right = left;
    right.arm = "so101_right";
    right.refined_base_in_camera = right.nominal_base_in_camera;
    right.translation_correction_m = 0.0;
    right.rotation_correction_rad = 0.0;
    estimate.mount_refinement = { left, right };
    estimate.joint_prior.enabled = true;
    estimate.joint_prior.status = "applied";
    estimate.joint_prior.age_ns = 0U; // OpenCV's test JSON parser does not support null.
    for (const auto* arm : { "so101_left", "so101_right" }) {
        for (int joint = 1; joint <= 6; ++joint) {
            estimate.joint_angles[arm][std::to_string(joint)] = 0.25;
            estimate.joint_prior.priors.push_back({ arm, std::to_string(joint), 0.25, 0.1 });
        }
    }
    const auto json = rbt_m3t::tracking_json(estimate, output_pair(), 1U, 0U);
    cv::FileStorage storage(json, cv::FileStorage::READ | cv::FileStorage::MEMORY
                                     | cv::FileStorage::FORMAT_JSON);
    const auto refinement = storage["mount_refinement"];
    check(static_cast<int>(refinement["enabled"]) == 1
              && static_cast<int>(refinement["joints_locked_to_feedback"]) == 1
              && refinement["arms"].size() == 2U,
          "mount refinement has enabled/feedback flags and both arms");
    const auto arm = refinement["arms"][0];
    check(static_cast<std::string>(arm["arm"]) == "so101_left"
              && arm["nominal_base_in_camera"].size() == 16U
              && arm["refined_base_in_camera"].size() == 16U
              && static_cast<double>(arm["nominal_base_in_camera"][3]) == 0.25
              && static_cast<double>(arm["refined_base_in_camera"][3])
                     == static_cast<double>(0.27F)
              && static_cast<double>(arm["translation_correction_m"]) == 0.02
              && static_cast<double>(arm["rotation_correction_rad"]) == 0.03,
          "mount refinement serializes row-major rigid poses and correction magnitudes");
    const auto priors = storage["joint_priors"]["priors"];
    check(static_cast<int>(storage["joint_priors"]["prior_count"]) == 12 && priors.size() == 12U,
          "mount refinement retains all twelve joint priors");
    for (int index = 0; index < 12; ++index) {
        check(static_cast<std::string>(priors[index]["arm"])
                      == (index < 6 ? "so101_left" : "so101_right")
                  && static_cast<std::string>(priors[index]["joint"])
                         == std::to_string(index % 6 + 1)
                  && static_cast<double>(priors[index]["angle_rad"]) == 0.25
                  && static_cast<double>(priors[index]["standard_deviation_rad"]) == 0.1,
              "each feedback prior retains its arm, joint, angle and sigma");
    }
    auto combined = estimate;
    combined.encoder_reference_links = true;
    combined.joints_locked_to_feedback = false;
    const auto combined_json = rbt_m3t::tracking_json(combined, output_pair(), 1U, 0U);
    cv::FileStorage combined_storage(combined_json,
        cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
    const auto combined_refinement = combined_storage["mount_refinement"];
    check(static_cast<int>(combined_storage["encoder_reference_links"]) == 1
              && static_cast<int>(combined_refinement["enabled"]) == 1
              && static_cast<int>(combined_refinement["joints_locked_to_feedback"]) == 0
              && combined_refinement["arms"].size() == 2U
              && combined_storage["joint_priors"]["priors"].size() == 12U,
          "combined output preserves mount diagnostics and all soft encoder priors with unlocked joints");
    const auto reject = [](const TrackingEstimate& invalid_estimate, const std::string& description) {
        expect_invalid([&] {
            (void)rbt_m3t::tracking_json(invalid_estimate, output_pair(), 1U, 0U);
        }, description);
    };
    auto bad = estimate;
    bad.mount_refinement.back().arm = "unknown_arm";
    reject(bad, "mount diagnostics cannot claim an arm absent from links and joint angles");
    for (const bool translation : { false, true }) {
        bad = estimate;
        if (translation) {
            bad.mount_refinement.front().translation_correction_m += 0.001;
        } else {
            bad.mount_refinement.front().rotation_correction_rad += 0.001;
        }
        reject(bad, "mount correction magnitudes must agree with nominal/refined poses");
    }
    bad = estimate;
    bad.joints_locked_to_feedback = false;
    reject(bad, "unlocked mount diagnostics require encoder reference links");
    bad = combined;
    bad.joints_locked_to_feedback = true;
    reject(bad, "encoder reference links cannot report locked joints in mount diagnostics");
    bad = estimate;
    bad.mount_refinement.clear();
    reject(bad, "feedback locking requires mount diagnostics");
    bad.encoder_reference_links = true;
    reject(bad, "feedback locking without diagnostics is invalid even with encoder references");
    bad = estimate;
    bad.mount_refinement.push_back(left);
    reject(bad, "mount diagnostics require unique arms");
    for (const auto* name : { "", "bad arm" }) {
        bad = estimate;
        bad.mount_refinement.front().arm = name;
        reject(bad, "mount diagnostics require valid arm names");
    }
    for (const double value : { -0.01, std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN() }) {
        bad = estimate;
        bad.mount_refinement.front().translation_correction_m = value;
        reject(bad, "mount translation correction must be finite and nonnegative");
        bad = estimate;
        bad.mount_refinement.front().rotation_correction_rad = value;
        reject(bad, "mount rotation correction must be finite and nonnegative");
    }
    for (const bool nominal : { false, true }) {
        for (const float value : { 2.0F, std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::quiet_NaN() }) {
            bad = estimate;
            auto& pose = nominal ? bad.mount_refinement.front().nominal_base_in_camera
                                 : bad.mount_refinement.front().refined_base_in_camera;
            pose.matrix()(0, 0) = value;
            reject(bad, "both mount poses must be finite rigid transforms");
        }
    }
}

void test_visual_link_weights_output_is_optional_and_strict() {
    TrackingEstimate estimate;
    estimate.links.push_back(LinkEstimate { "so101_left", "gripper", Eigen::Affine3f::Identity() });
    check(rbt_m3t::tracking_json(estimate, output_pair(), 1U, 0U).find("visual_link_weights")
              == std::string::npos,
          "legacy estimates omit visual link weights");
    estimate.visual_link_weights = { { "gripper", 8.0 }, { "wrist", 0.5 } };
    const auto output = rbt_m3t::tracking_json(estimate, output_pair(), 1U, 0U);
    check(output.find("\"visual_link_weights\":{\"gripper\":8,\"wrist\":0.5}")
              != std::string::npos,
          "visual link weights serialize as a top-level object with exact multipliers");
    check(output.find("mount_refinement") == std::string::npos,
          "visual weights do not introduce mount refinement diagnostics");
    for (const auto* link : { "", "bad name", "arm/gripper" }) {
        estimate.visual_link_weights = { { link, 1.0 } };
        expect_invalid([&] { (void)rbt_m3t::tracking_json(estimate, output_pair(), 1U, 0U); },
                       "visual weight output rejects invalid link names");
    }
    estimate.visual_link_weights = { { std::string(257U, 'x'), 1.0 } };
    expect_invalid([&] { (void)rbt_m3t::tracking_json(estimate, output_pair(), 1U, 0U); },
                   "visual weight output rejects overlong link names");
    for (const double weight : { 0.0, -1.0, 0.0099, 100.001,
                                 std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN() }) {
        estimate.visual_link_weights = { { "gripper", weight } };
        expect_invalid([&] { (void)rbt_m3t::tracking_json(estimate, output_pair(), 1U, 0U); },
                       "visual weight output rejects nonfinite and out-of-range multipliers");
    }
}

void test_serializer_rejects_nonrigid_and_nonfinite_poses() {
    TrackingEstimate estimate;
    auto pose = Eigen::Affine3f::Identity();
    pose.matrix()(0, 0) = 2.0F;
    estimate.links.push_back(LinkEstimate { "so101_left", "tip", pose });
    expect_invalid([&] { (void)rbt_m3t::tracking_json(estimate, output_pair(), 0U, 0U); },
                   "non-rigid estimate pose");

    pose = Eigen::Affine3f::Identity();
    pose.matrix()(0, 0) = std::numeric_limits<float>::quiet_NaN();
    estimate.links[0].in_camera = pose;
    expect_invalid([&] { (void)rbt_m3t::tracking_json(estimate, output_pair(), 0U, 0U); },
                   "non-finite estimate pose");
}

void test_encoder_reference_links_require_complete_feedback() {
    Fixture fixture;
    const auto load = [&](const std::string& extensions) {
        Fixture::write(fixture.config_path(), Fixture::config_json(
            "200", "../initialization.json", extensions));
        return rbt_m3t::load_tracking_config(fixture.config_path());
    };
    const auto left = Fixture::joint_prior_mapping();
    const auto both = left + "," + Fixture::joint_prior_mapping("so101_right", "5", "right");
    const auto priors = [](const std::string& mappings, const std::string& extra = "") {
        return Fixture::joint_prior_extension(mappings,
            "\"service_name\":\"joint-state\",\"controller_id\":\"controller-1\","
            "\"required\":true," + extra);
    };
    const std::string fusion = ",\"encoder_reference_links\":true";
    check(!load("").engine.encoder_reference_links, "encoder reference links are opt-in");
    check(load(fusion + priors(both)).engine.encoder_reference_links,
          "encoder reference links accept all required joint measurements");
    const auto nominal_path = fixture.root() / "nominal.json";
    const auto nominal = Fixture::initialization_json("-1.37", "4.4906552",
        "[1,0,0,0.35,0,1,0,0,0,0,1,0,0,0,0,1]");
    Fixture::write(nominal_path, nominal);
    const std::string tf_nominal = ",\"tf_nominal_initialization\":\"../nominal.json\"";
    const auto with_anchor = load(fusion + priors(both) + tf_nominal);
    check(with_anchor.tf_nominal_arms.size() == with_anchor.arms.size()
              && std::abs(with_anchor.tf_nominal_arms.front().base_in_camera.translation().x()
                          - 0.35F) < 1.0e-6F,
          "TF can use an explicit nominal CAD frame separate from the tracking base");
    expect_invalid([&] { (void)load(priors(both) + tf_nominal); },
                   "nominal TF anchors require encoder reference links");
    auto wrong_camera = nominal;
    const auto coordinate = wrong_camera.find("camera/color_optical");
    check(coordinate != std::string::npos, "nominal fixture has a camera frame");
    wrong_camera.replace(coordinate, std::string("camera/color_optical").size(), "other/color_optical");
    Fixture::write(nominal_path, wrong_camera);
    expect_invalid([&] { (void)load(fusion + priors(both) + tf_nominal); },
                   "nominal anchors must match the tracking camera");
    expect_invalid([&] { (void)load(fusion); }, "encoder reference links require feedback");
    expect_invalid([&] { (void)load(fusion + priors(left)); },
                   "encoder reference links require both arms");
    expect_invalid([&] { (void)load(fusion + priors(both, "\"initialize_from_measurement\":false,")); },
                   "encoder reference links require measured initialization");
    expect_invalid([&] {
        (void)load(fusion + priors(both) + ",\"mount_refinement\":{\"enabled\":true}");
    }, "combined mount refinement rejects a fixed base");
    expect_invalid([&] {
        (void)load(fusion + priors(both) + ",\"initialization_flow\":{\"refine\":true}");
    }, "encoder reference links reject initialization refinement");
}

} // namespace

int main() {
    try {
        test_load_relative_files_and_exact_joint_seeds();
        test_initialization_accepts_one_arm_and_rejects_bad_cardinality();
        test_secondary_camera_config_is_optional_and_strict();
        test_camera_motion_config_is_opt_in_and_strict();
        test_reject_missing_malformed_and_stale_config();
        test_reject_missing_init_and_invalid_physical_seeds();
        test_load_nested_options_metadata_and_constraints();
        test_reject_invalid_nested_options_constraints_and_metadata_paths();
        test_joint_prior_config_is_explicit_and_bounded();
        test_visual_link_weights_config_is_optional_and_bounded();
        test_mount_refinement_config_requires_complete_feedback_and_static_camera();
        test_encoder_reference_links_require_complete_feedback();
        test_serializer_preserves_integer_timing_and_row_major_pose();
        test_mount_refinement_output_is_strict_and_preserves_all_joint_priors();
        test_visual_link_weights_output_is_optional_and_strict();
        test_serializer_rejects_nonrigid_and_nonfinite_poses();
    } catch (const std::exception& error) {
        std::cerr << "M3T app config test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "M3T app config tests passed\n";
    return 0;
}
