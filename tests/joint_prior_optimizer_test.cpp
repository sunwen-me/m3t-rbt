#include <rbt_m3t/model.hpp>
#include <rbt_m3t/weighted_modality.hpp>

#include <m3t/link.h>
#include <m3t/modality.h>
#include <m3t/refiner.h>
#include <m3t/optimizer.h>
#include <m3t/soft_constraint.h>
#include <m3t/static_detector.h>

#include <Eigen/Geometry>

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string &description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

const std::array<bool, 6> kZRotationOnly{false, false, true,
                                         false, false, false};
const std::array<bool, 6> kFixed{false, false, false,
                                 false, false, false};

float rotation_z(const m3t::Transform3fA &pose) {
  return std::atan2(pose.rotation()(1, 0), pose.rotation()(0, 0));
}

m3t::Transform3fA rotation_pose_z(float angle) {
  auto pose = m3t::Transform3fA::Identity();
  pose.linear() = Eigen::AngleAxisf{angle, Eigen::Vector3f::UnitZ()}
                      .toRotationMatrix();
  return pose;
}

struct OptimizerFixture {
  OptimizerFixture(bool free_base, float child_angle = 0.0f,
                   bool add_soft_constraint = false)
      : root{std::make_shared<m3t::Link>("base", nullptr,
                                         m3t::Transform3fA::Identity(),
                                         m3t::Transform3fA::Identity(),
                                         m3t::Transform3fA::Identity(),
                                         free_base ? kZRotationOnly : kFixed)},
        parent{std::make_shared<m3t::Link>(
            "parent", nullptr, m3t::Transform3fA::Identity(),
            m3t::Transform3fA::Identity(), m3t::Transform3fA::Identity(),
            kZRotationOnly)},
        child{std::make_shared<m3t::Link>(
            "child", nullptr, m3t::Transform3fA::Identity(),
            rotation_pose_z(child_angle),
            m3t::Transform3fA::Identity(), kZRotationOnly)},
        optimizer{"joint prior fixture", root, 1.0f, 1.0f} {
    check(root->AddChildLink(parent), "fixture adds parent link");
    check(parent->AddChildLink(child), "fixture adds child link");
    check(root->SetUp(), "fixture sets up base link");
    check(parent->SetUp(), "fixture sets up parent link");
    check(child->SetUp(), "fixture sets up child link");

    if (add_soft_constraint) {
      soft_constraint = std::make_shared<m3t::SoftConstraint>(
          "joint relation", parent, child, m3t::Transform3fA::Identity(),
          m3t::Transform3fA::Identity(), kZRotationOnly, 0.0f, 0.0f, 0.5f,
          0.01f);
      check(soft_constraint->SetUp(), "fixture sets up soft constraint");
      check(optimizer.AddSoftConstraint(soft_constraint),
            "fixture adds soft constraint");
    }
    check(optimizer.SetUp(), "fixture sets up optimizer without OpenGL");
  }

  int child_index() const { return child->first_jacobian_index(); }

  float base_angle() const { return rotation_z(root->link2world_pose()); }

  float parent_angle() const { return rotation_z(parent->joint2parent_pose()); }

  float child_angle() const { return rotation_z(child->joint2parent_pose()); }

  std::shared_ptr<m3t::Link> root;
  std::shared_ptr<m3t::Link> parent;
  std::shared_ptr<m3t::Link> child;
  std::shared_ptr<m3t::SoftConstraint> soft_constraint;
  m3t::Optimizer optimizer;
};

// A real modality is needed for native Refiner to select its optimizer.
// Zero image information isolates the scalar prior while exercising the full
// correspondence, optimization, and visualization pipeline without OpenGL.
class CpuModality : public m3t::Modality {
 public:
  explicit CpuModality(const std::shared_ptr<m3t::Body>& body)
      : Modality{"cpu modality", body} {}

  bool SetUp() override {
    gradient_.setZero();
    hessian_.setZero();
    set_up_ = true;
    return true;
  }
  bool StartModality(int iteration, int corr_iteration) override {
    check(iteration == 0 && corr_iteration == starts,
          "native refiner restarts the modality at each correspondence iteration");
    ++starts;
    return true;
  }
  bool CalculateCorrespondences(int, int) override {
    ++correspondences;
    return true;
  }
  bool VisualizeCorrespondences(int) override {
    ++correspondence_visualizations;
    return true;
  }
  bool CalculateGradientAndHessian(int, int, int) override {
    ++gradients;
    return true;
  }
  bool VisualizeOptimization(int) override {
    ++optimization_visualizations;
    return true;
  }
  bool CalculateResults(int) override { return true; }
  bool VisualizeResults(int) override { return true; }
  std::vector<std::shared_ptr<m3t::Camera>> camera_ptrs() const override {
    return {};
  }

  void reset_counts() {
    starts = correspondences = correspondence_visualizations = gradients =
        optimization_visualizations = 0;
  }

  int starts = 0;
  int correspondences = 0;
  int correspondence_visualizations = 0;
  int gradients = 0;
  int optimization_visualizations = 0;
};

// A deterministic visual objective, reset on every calculation like native modalities.
class AnalyticVisualModality : public CpuModality {
 public:
  using CpuModality::CpuModality;

  bool CalculateGradientAndHessian(int iteration, int corr_iteration,
                                   int opt_iteration) override {
    if (!CpuModality::CalculateGradientAndHessian(iteration, corr_iteration, opt_iteration))
      return false;
    gradient_ << 1.5f, -2.0f, 3.5f, 2.4f, -5.0f, 6.0f;
    hessian_ = -2.0f * Eigen::Matrix<float, 6, 6>::Identity();
    hessian_(3, 3) = -4.0f;
    hessian_(0, 3) = hessian_(3, 0) = -0.5f;
    return !fail_calculation;
  }

  bool fail_calculation = false;
};

void visual_weights_scale_the_entire_objective_and_preserve_cad_prior() {
  AnalyticVisualModality unweighted{nullptr};
  rbt_m3t::WeightedVisualModality<AnalyticVisualModality> weighted{nullptr};
  check(unweighted.SetUp() && weighted.SetUp(), "synthetic weighted modalities set up on CPU");
  check(unweighted.CalculateGradientAndHessian(0, 0, 0)
            && weighted.CalculateGradientAndHessian(0, 0, 0),
        "default visual weight calculates the base objective");
  check((weighted.gradient().array() == unweighted.gradient().array()).all()
            && (weighted.hessian().array() == unweighted.hessian().array()).all(),
        "default weight one leaves every gradient and Hessian coefficient unchanged");

  double previous_solution = 0.0;
  for (const double weight : {0.01, 0.25, 1.0, 4.0, 100.0}) {
    weighted.set_visual_weight(weight);
    for (int update = 0; update < 2; ++update) {
      check(weighted.CalculateGradientAndHessian(0, 0, update),
            "weighted visual objective recalculates successfully");
      const float scale = static_cast<float>(weight);
      check(weighted.gradient().isApprox(unweighted.gradient() * scale, 1.0e-6f)
                && weighted.hessian().isApprox(unweighted.hessian() * scale, 1.0e-6f),
            "weight scales both gradient and full Hessian, without accumulating between updates");
    }
    // Isolate translation x: E = .5*w*4*(x-.6)^2 + .5*(x/.05)^2.
    // The fixed CAD prior stays at precision 400; only the visual term is scaled.
    const double solution = weighted.gradient()(3) / (400.0 - weighted.hessian()(3, 3));
    const double expected = (weight * 4.0 * 0.6) / (400.0 + weight * 4.0);
    check(std::abs(solution - expected) < 1.0e-7,
          "weighted visual plus unchanged CAD prior has the expected analytic solution");
    check(solution > previous_solution,
          "increasing visual weight moves the solution toward the visual target");
    previous_solution = solution;
  }
  weighted.set_visual_weight(0.25);
  for (const double invalid : {0.0, -1.0, 0.0099, 100.0001,
           std::numeric_limits<double>::infinity(),
           -std::numeric_limits<double>::infinity(),
           std::numeric_limits<double>::quiet_NaN()}) {
    bool rejected = false;
    try {
      weighted.set_visual_weight(invalid);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    check(rejected, "visual wrapper rejects non-finite and out-of-range weights");
    check(weighted.CalculateGradientAndHessian(0, 0, 0)
              && weighted.gradient().isApprox(unweighted.gradient() * 0.25f, 1.0e-6f)
              && weighted.hessian().isApprox(unweighted.hessian() * 0.25f, 1.0e-6f),
          "invalid weight leaves the previous valid objective weight intact");
  }
  weighted.fail_calculation = true;
  check(!weighted.CalculateGradientAndHessian(0, 0, 0),
        "wrapper propagates native gradient calculation failure");
  check(weighted.gradient().isApprox(unweighted.gradient(), 1.0e-6f)
            && weighted.hessian().isApprox(unweighted.hessian(), 1.0e-6f),
        "wrapper does not scale a failed native calculation");
}

void engine_rejects_invalid_visual_link_weights_before_setup() {
  rbt_m3t::RobotModel model;
  model.name = "visual weight configuration";
  model.root_link = "visual";
  model.links = {{"visual", "unused_weight_fixture.obj"}, {"virtual", {}}};
  rbt_m3t::ModelJoint joint;
  joint.name = "fixed";
  joint.type = "fixed";
  joint.parent = "visual";
  joint.child = "virtual";
  model.joints.push_back(joint);
  rbt_m3t::ArmSeed seed;
  seed.name = "arm";
  rbt_m3t::EngineOptions options;
  options.cache_directory = std::filesystem::temp_directory_path() / "unused_visual_weight_cache";
  auto images = std::make_shared<rbt_m3t::CameraImages>();
  const auto reject = [&] {
    bool rejected = false;
    try {
      rbt_m3t::TrackerEngine engine{model, {seed}, options, images};
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    check(rejected, "engine rejects invalid visual weights before creating bodies or model caches");
  };
  for (const double weight : {0.0, 0.0099, 100.0001,
           std::numeric_limits<double>::infinity(),
           std::numeric_limits<double>::quiet_NaN()}) {
    options.visual_link_weights = {{"visual", weight}};
    reject();
  }
  options.visual_link_weights = {{"missing", 1.0}};
  reject();
  options.visual_link_weights = {{"virtual", 1.0}};
  reject();
  for (const double weight : {0.01, 1.0, 100.0}) {
    options.visual_link_weights = {{"visual", weight}};
    rbt_m3t::TrackerEngine engine{model, {seed}, options, images};
  }
  options.visual_link_weights.clear();
  rbt_m3t::TrackerEngine legacy{model, {seed}, options, images};
}

struct RefinerFixture {
  RefinerFixture() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    directory = std::filesystem::temp_directory_path()
                / ("rbt_m3t_refiner_test_" + std::to_string(nonce));
    std::filesystem::create_directories(directory);
    const auto mesh = directory / "triangle.obj";
    std::ofstream{mesh} << "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nf 1 2 3\n";
    const auto metadata = directory / "refiner.yaml";
    std::ofstream{metadata} << "%YAML:1.0\n"
                              "n_corr_iterations: 2\nn_update_iterations: 3\n"
                              "visualization_time: 1\n";
    body = std::make_shared<m3t::Body>(
        "cpu body", mesh, 1.0f, true, false, m3t::Transform3fA::Identity());
    modality = std::make_shared<CpuModality>(body);
    root = std::make_shared<m3t::Link>(
        "root", nullptr, m3t::Transform3fA::Identity(),
        m3t::Transform3fA::Identity(), m3t::Transform3fA::Identity(), kFixed);
    joint = std::make_shared<m3t::Link>(
        "joint", body, m3t::Transform3fA::Identity(),
        m3t::Transform3fA::Identity(), m3t::Transform3fA::Identity(), kZRotationOnly);
    check(joint->AddModality(modality), "CPU fixture attaches its modality");
    check(root->AddChildLink(joint), "CPU fixture attaches its revolute joint");
    optimizer = std::make_shared<m3t::Optimizer>("cpu optimizer", root, 1.0f, 1.0f);
    refiner = std::make_shared<m3t::Refiner>("cpu refiner", metadata);
    check(refiner->AddOptimizer(optimizer), "CPU fixture attaches its optimizer");
    check(refiner->SetUp(), "CPU fixture sets up native refiner without OpenGL");
    check(refiner->n_corr_iterations() == 2 && refiner->n_update_iterations() == 3
              && refiner->visualization_time() == 1,
          "refiner retains iteration counts and visualization time from metadata");
  }
  ~RefinerFixture() { std::filesystem::remove_all(directory); }

  float angle() const { return rotation_z(joint->joint2parent_pose()); }
  bool install_prior(float current) {
    return optimizer->SetJointAnglePriors(
        {{joint->first_jacobian_index(), current, 0.4f, 0.1f}});
  }
  bool refine() { return refiner->RefinePoses({optimizer->name()}); }

  std::filesystem::path directory;
  std::shared_ptr<m3t::Body> body;
  std::shared_ptr<CpuModality> modality;
  std::shared_ptr<m3t::Link> root;
  std::shared_ptr<m3t::Link> joint;
  std::shared_ptr<m3t::Optimizer> optimizer;
  std::shared_ptr<m3t::Refiner> refiner;
};

void fixed_cad_anchor_constrains_only_the_six_dof_base() {
  const std::array<bool, 6> all{true, true, true, true, true, true};
  const auto identity = m3t::Transform3fA::Identity();
  auto nominal = rotation_pose_z(0.47f);
  nominal.translation() = Eigen::Vector3f{0.4f, -0.3f, 0.8f};
  auto anchor = std::make_shared<m3t::Link>(
      "CAD anchor", nullptr, identity, identity, nominal, kFixed, true);
  auto base = std::make_shared<m3t::Link>(
      "physical base", nullptr, identity, identity, nominal, all, true);
  auto joint_pose = rotation_pose_z(0.7123457f);
  joint_pose.translation() = Eigen::Vector3f{0.1f, 0.05f, 0.2f};
  auto joint = std::make_shared<m3t::Link>(
      "encoder joint", nullptr, identity, joint_pose, identity, kFixed, true);
  const auto gripper_pose = rotation_pose_z(-0.2345679f);
  auto gripper = std::make_shared<m3t::Link>(
      "encoder gripper", nullptr, identity, gripper_pose, identity, kFixed, true);
  check(anchor->AddChildLink(base) && base->AddChildLink(joint)
            && joint->AddChildLink(gripper),
        "CAD fixture builds a fixed virtual anchor, free base, and locked encoder descendants");
  check(anchor->SetUp() && base->SetUp() && joint->SetUp() && gripper->SetUp(),
        "CAD fixture sets up without a Body or OpenGL");
  auto constraint = std::make_shared<m3t::SoftConstraint>(
      "CAD mount", anchor, base, identity, identity, all, 0.0f, 0.0f, 0.08f, 0.02f);
  check(constraint->SetUp(), "CAD fixture sets up the six-axis zero-deadzone soft prior");
  auto optimizer = std::make_shared<m3t::Optimizer>("CAD optimizer", anchor, 1.0f, 1.0f);
  check(optimizer->AddSoftConstraint(constraint) && optimizer->SetUp(),
        "optimizer accepts a fixed body-free root with a six-DoF physical child");
  check(anchor->DegreesOfFreedom() == 0 && base->DegreesOfFreedom() == 6
            && joint->DegreesOfFreedom() == 0 && gripper->DegreesOfFreedom() == 0,
        "only the mount correction owns optimizer coordinates");
  check(base->link2world_pose().matrix().isApprox(nominal.matrix(), 1.0e-6f),
        "nonidentity CAD reference is applied exactly once at setup");

  // Apply a six-axis optimizer increment without introducing any scalar priors.
  Eigen::VectorXf delta(6);
  delta << 0.03f, -0.02f, 0.04f, 0.025f, -0.015f, 0.01f;
  check(base->UpdatePoses(anchor, delta) && optimizer->CalculateConsistentPoses(),
        "six-axis base correction propagates through locked descendants");
  const auto displaced = base->joint2parent_pose();
  const float initial_translation = displaced.translation().norm();
  const float initial_rotation = Eigen::AngleAxisf(displaced.rotation()).angle();
  check(initial_translation > 0.02f && initial_rotation > 0.04f,
        "physical base remains movable in both rotation and translation");
  check(anchor->link2world_pose().matrix().isApprox(nominal.matrix(), 1.0e-6f),
        "base variation never moves the CAD anchor");

  // A new frame changes encoder transforms, retaining the previous base delta.
  joint_pose = rotation_pose_z(-0.413579f);
  joint_pose.translation() = Eigen::Vector3f{0.1f, 0.05f, 0.2f};
  joint->set_joint2parent_pose(joint_pose);
  check(optimizer->CalculateConsistentPoses(), "fresh encoder transform propagates without setup");
  check(base->joint2parent_pose().matrix().isApprox(displaced.matrix(), 1.0e-6f),
        "fresh encoder feedback preserves the mount correction across frames");
  check(gripper->link2world_pose().matrix().isApprox(
            (nominal * displaced * joint_pose * gripper_pose).matrix(), 1.0e-6f),
        "descendant world pose combines CAD, correction and encoder transforms once");
  for (int update = 0; update < 6; ++update) {
    check(optimizer->CalculateOptimization(0, 0, update), "soft CAD prior optimizes on CPU");
    check(anchor->link2world_pose().matrix().isApprox(nominal.matrix(), 1.0e-6f),
          "CAD anchor stays fixed during every optimization update");
    check((joint->joint2parent_pose().matrix().array() == joint_pose.matrix().array()).all()
              && (gripper->joint2parent_pose().matrix().array() == gripper_pose.matrix().array()).all(),
          "optimization cannot drift fixed encoder or gripper joint transforms");
  }
  check(base->joint2parent_pose().translation().norm() < initial_translation * 0.01f
            && Eigen::AngleAxisf(base->joint2parent_pose().rotation()).angle()
                   < initial_rotation * 0.01f,
        "soft constraint pulls the displaced physical base back to its CAD mount");

  check(base->UpdatePoses(anchor, delta), "displace base again before explicit reset");
  auto replacement_nominal = rotation_pose_z(-0.61f);
  replacement_nominal.translation() = Eigen::Vector3f{-0.2f, 0.6f, 0.9f};
  m3t::StaticDetector reset{"CAD reset", optimizer, replacement_nominal, false};
  check(reset.SetUp(), "internal detector remains available for explicit anchor reset");
  base->set_joint2parent_pose(identity);
  joint->set_joint2parent_pose(joint_pose);
  check(reset.DetectPoses({optimizer->name()}, nullptr), "reset updates the optimizer's CAD root");
  check(anchor->link2world_pose().matrix().isApprox(replacement_nominal.matrix(), 1.0e-6f)
            && base->link2world_pose().matrix().isApprox(replacement_nominal.matrix(), 1.0e-6f),
        "explicit reset clears the old delta and replaces the CAD reference without double multiplication");
  check(gripper->link2world_pose().matrix().isApprox(
            (replacement_nominal * joint_pose * gripper_pose).matrix(), 1.0e-6f),
        "reset preserves the newest encoder chain under the replacement CAD mount");
}

void soft_cad_mount_and_encoder_reference_joint_optimize_together() {
  const std::array<bool, 6> all{true, true, true, true, true, true};
  const std::array<bool, 6> translation_only{false, false, false, true, true, true};
  const auto identity = m3t::Transform3fA::Identity();
  auto nominal = rotation_pose_z(0.47f);
  nominal.translation() = Eigen::Vector3f{0.4f, -0.3f, 0.8f};
  float encoder_angle = 0.6f;
  constexpr float visual_angle = 0.1f;
  const Eigen::Vector3f visual_translation{0.04f, -0.02f, 0.03f};
  auto anchor = std::make_shared<m3t::Link>(
      "combined CAD anchor", nullptr, identity, identity, nominal, kFixed, true);
  auto base = std::make_shared<m3t::Link>(
      "combined base", nullptr, identity, identity, nominal, all, true);
  auto reference = std::make_shared<m3t::Link>(
      "combined encoder reference", nullptr, identity, rotation_pose_z(encoder_angle),
      identity, kFixed, true);
  auto joint = std::make_shared<m3t::Link>(
      "combined physical joint", nullptr, identity, identity, identity, kZRotationOnly, true);
  check(anchor->AddChildLink(base) && base->AddChildLink(reference)
            && reference->AddChildLink(joint),
        "combined graph attaches a six-DoF base and a one-DoF encoder correction");
  check(anchor->SetUp() && base->SetUp() && reference->SetUp() && joint->SetUp(),
        "combined graph sets up all physical and reference links on CPU");
  auto mount_prior = std::make_shared<m3t::SoftConstraint>(
      "combined mount prior", anchor, base, identity, identity, all,
      0.0f, 0.0f, 0.08726646f, 0.02f);
  auto encoder_prior = std::make_shared<m3t::SoftConstraint>(
      "combined encoder prior", reference, joint, identity, identity, kZRotationOnly,
      0.0f, 0.0f, 0.3f, 0.01f);
  // These two deterministic factors replace visual evidence. The orientation
  // factor couples base yaw and joint correction; the translation factor pulls
  // the base along all three translation coordinates.
  auto orientation_evidence = std::make_shared<m3t::SoftConstraint>(
      "combined visual orientation", anchor, joint, rotation_pose_z(-visual_angle),
      identity, kZRotationOnly, 0.0f, 0.0f, 0.1f, 0.01f);
  auto translation_target = identity;
  translation_target.translation() = -visual_translation;
  auto translation_evidence = std::make_shared<m3t::SoftConstraint>(
      "combined visual translation", anchor, base, translation_target, identity,
      translation_only, 0.0f, 0.0f, 0.1f, 0.01f);
  check(mount_prior->SetUp() && encoder_prior->SetUp()
            && orientation_evidence->SetUp() && translation_evidence->SetUp(),
        "combined graph sets up both soft priors and opposing visual evidence");
  m3t::Optimizer optimizer{"combined optimizer", anchor, 1.0f, 1.0f};
  check(optimizer.AddSoftConstraint(mount_prior) && optimizer.AddSoftConstraint(encoder_prior)
            && optimizer.AddSoftConstraint(orientation_evidence)
            && optimizer.AddSoftConstraint(translation_evidence) && optimizer.SetUp(),
        "combined optimizer includes mount and encoder priors in the same solve");
  check(anchor->DegreesOfFreedom() == 0 && base->DegreesOfFreedom() == 6
            && reference->DegreesOfFreedom() == 0 && joint->DegreesOfFreedom() == 1
            && joint->first_jacobian_index() == 6,
        "combined graph retains six mount coordinates and the joint correction coordinate");

  Eigen::VectorXf initial_delta(7);
  initial_delta << 0.04f, -0.03f, 0.02f, 0.01f, -0.015f, 0.005f, 0.08f;
  check(base->UpdatePoses(anchor, initial_delta)
            && joint->UpdatePoses(reference, initial_delta) && optimizer.CalculateConsistentPoses(),
        "combined graph accepts rotation, translation and joint optimizer increments");
  check(base->joint2parent_pose().translation().norm() > 0.01f
            && Eigen::AngleAxisf(base->joint2parent_pose().rotation()).angle() > 0.04f
            && std::abs(rotation_z(joint->joint2parent_pose()) - 0.08f) < 1.0e-6f,
        "combined physical base and encoder correction can both leave their references");

  struct Solution {
    float base_angle;
    float joint_angle;
    Eigen::Vector3f translation;
  };
  const auto solve = [&](float mount_rotation_sigma, float mount_translation_sigma,
                          float encoder_sigma) {
    mount_prior->set_standard_deviation_rotation(mount_rotation_sigma);
    mount_prior->set_standard_deviation_translation(mount_translation_sigma);
    encoder_prior->set_standard_deviation_rotation(encoder_sigma);
    // A strong relative prior adds curvature to the shared base coordinates;
    // allow convergence instead of assuming every sigma settles in a few steps.
    int stable_updates = 0;
    for (int update = 0; update < 2400; ++update) {
      const auto previous_base = base->joint2parent_pose().matrix().eval();
      const float previous_joint = rotation_z(joint->joint2parent_pose());
      check(optimizer.CalculateOptimization(0, 0, update),
            "combined mount and joint optimization executes on CPU");
      auto correction = base->joint2parent_pose();
      Eigen::Quaternionf rotation{correction.linear()};
      rotation.normalize();
      correction.linear() = rotation.toRotationMatrix();
      base->set_joint2parent_pose(correction);
      joint->set_joint2parent_pose(rotation_pose_z(rotation_z(joint->joint2parent_pose())));
      check(optimizer.CalculateConsistentPoses(), "combined projected coordinates propagate consistently");
      check(anchor->link2world_pose().matrix().isApprox(nominal.matrix(), 1.0e-6f),
            "combined optimization retains its fixed nominal CAD anchor");
      const auto base_rotation = base->link2world_pose().linear();
      check((base_rotation.transpose() * base_rotation).isApprox(Eigen::Matrix3f::Identity(), 1.0e-6f)
                && std::abs(base_rotation.determinant() - 1.0f) < 1.0e-6f,
            "combined mount projection retains a proper rigid base rotation");
      const float base_step = (base->joint2parent_pose().matrix() - previous_base)
                                  .cwiseAbs().maxCoeff();
      const float joint_step = std::abs(rotation_z(joint->joint2parent_pose()) - previous_joint);
      stable_updates = base_step < 1.0e-7f && joint_step < 1.0e-7f ? stable_updates + 1 : 0;
      if (stable_updates == 4) break;
    }
    Solution solution{rotation_z(base->joint2parent_pose()),
                      encoder_angle + rotation_z(joint->joint2parent_pose()),
                      base->joint2parent_pose().translation()};
    // E = .5*b^2/s_b^2 + .5*c^2/s_q^2
    //   + .5*(b+encoder+c-visual)^2/s_v^2, with q=encoder+c.
    const float visual_variance = 0.1f * 0.1f;
    const float mount_variance = mount_rotation_sigma * mount_rotation_sigma;
    const float encoder_variance = encoder_sigma * encoder_sigma;
    const float residual = (encoder_angle - visual_angle)
                           / (1.0f + (mount_variance + encoder_variance) / visual_variance);
    const float expected_base = -mount_variance / visual_variance * residual;
    const float expected_joint = encoder_angle - encoder_variance / visual_variance * residual;
    const float translation_variance = mount_translation_sigma * mount_translation_sigma;
    const Eigen::Vector3f expected_translation =
        translation_variance / (translation_variance + 0.01f * 0.01f) * visual_translation;
    check(std::abs(solution.base_angle - expected_base) < 2.0e-4f
              && std::abs(solution.joint_angle - expected_joint) < 2.0e-4f
              && solution.translation.isApprox(expected_translation, 1.0e-4f),
          "combined visual and soft-prior equilibrium matches the analytic coupled solution: base="
              + std::to_string(solution.base_angle) + "/" + std::to_string(expected_base)
              + " joint=" + std::to_string(solution.joint_angle) + "/" + std::to_string(expected_joint)
              + " translation error=" + std::to_string((solution.translation - expected_translation).norm()));
    return solution;
  };

  const auto baseline = solve(0.08726646f, 0.02f, 0.3f);
  check(std::abs(baseline.base_angle) > 0.01f && baseline.translation.norm() > 0.02f
            && std::abs(baseline.joint_angle - encoder_angle) > 0.1f,
        "visual evidence moves both the soft base and the soft encoder joint");
  const auto strong_mount = solve(0.01f, 0.005f, 0.3f);
  check(std::abs(strong_mount.base_angle) < std::abs(baseline.base_angle) * 0.1f
            && strong_mount.translation.norm() < baseline.translation.norm() * 0.3f
            && strong_mount.joint_angle < baseline.joint_angle - 0.01f,
        "a stronger mount prior changes the coupled equilibrium while leaving the joint free");
  const auto strong_encoder = solve(0.08726646f, 0.02f, 0.01f);
  check(std::abs(strong_encoder.joint_angle - encoder_angle) < 0.01f
            && std::abs(strong_encoder.base_angle) > std::abs(baseline.base_angle) * 2.0f,
        "a stronger encoder prior changes the coupled equilibrium while leaving the base free");

  nominal = rotation_pose_z(-0.31f);
  nominal.translation() = Eigen::Vector3f{-0.2f, 0.5f, 0.9f};
  anchor->set_link2world_pose(nominal);
  base->set_joint2parent_pose(identity);
  joint->set_joint2parent_pose(identity);
  encoder_angle = -0.4f;
  reference->set_joint2parent_pose(rotation_pose_z(encoder_angle));
  check(optimizer.CalculateConsistentPoses(), "combined reset replaces the CAD and encoder references");
  const auto reset = solve(0.08726646f, 0.02f, 0.3f);
  check(reset.base_angle > 0.01f && reset.joint_angle > encoder_angle + 0.1f,
        "both soft priors remain active after replacing their reference poses");
}

void native_refiner_refreshes_prior_at_every_update() {
  RefinerFixture baseline;
  check(baseline.refine(), "native refiner runs with unset hooks");
  check(std::abs(baseline.angle()) < 1.0e-6f,
        "unset hooks retain the no-prior baseline");

  RefinerFixture stale;
  check(stale.install_prior(stale.angle()), "installs one stale residual");
  check(stale.refine(), "native refiner runs with a stale residual");
  check(stale.angle() > 2.0f,
        "installing once overshoots the 0.4-radian target over six updates");

  RefinerFixture refreshed;
  std::vector<float> current_angles;
  int after_calls = 0;
  refreshed.refiner->set_before_optimization_callback([&] {
    current_angles.push_back(refreshed.angle());
    check(refreshed.modality->gradients == static_cast<int>(current_angles.size()),
          "before callback runs after each native gradient calculation");
    return refreshed.install_prior(refreshed.angle());
  });
  refreshed.refiner->set_after_optimization_callback([&] {
    ++after_calls;
    check(refreshed.modality->optimization_visualizations == after_calls - 1,
          "after callback runs before native optimization visualization");
    check(refreshed.body->body2world_pose().matrix().isApprox(
              (refreshed.root->link2world_pose()
               * refreshed.joint->joint2parent_pose()).matrix(), 1.0e-6f),
          "optimized body and link poses remain consistent");
    return refreshed.refiner->CalculateConsistentPoses();
  });
  check(refreshed.refiner->RefinePoses({"unselected"}),
        "unselected optimizer retains native no-op detection behavior");
  check(current_angles.empty() && after_calls == 0,
        "no callbacks run for unselected refinement");
  check(refreshed.refine(), "native refiner runs with refreshed residuals");
  check(current_angles.size() == 6U && after_calls == 6,
        "hooks run at all six native optimization boundaries");
  check(refreshed.modality->starts == 2 && refreshed.modality->correspondences == 2
            && refreshed.modality->correspondence_visualizations == 2
            && refreshed.modality->optimization_visualizations == 6,
        "hooks preserve native modality restart and visualization counts");
  check(current_angles.size() == 6U && current_angles.front() == 0.0f
            && current_angles[1] > 0.39f && current_angles.back() > 0.399f,
        "later updates see the current angle rather than the original residual");
  check(std::abs(refreshed.angle() - 0.4f) < 1.0e-5f,
        "refreshed residual converges to the target without overshoot");

  refreshed.refiner->set_before_optimization_callback({});
  refreshed.refiner->set_after_optimization_callback({});
  refreshed.optimizer->ClearJointAnglePriors();
  refreshed.joint->ResetJointPoses();
  refreshed.modality->reset_counts();
  check(refreshed.refine(), "refiner can be reused after clearing hooks and priors");
  check(current_angles.size() == 6U && after_calls == 6
            && std::abs(refreshed.angle()) < 1.0e-6f,
        "reset hooks and priors do not leak into subsequent refinement");
}

void native_refiner_propagates_hook_failures_and_allows_reset() {
  RefinerFixture fixture;
  int before_calls = 0;
  int after_calls = 0;
  fixture.refiner->set_before_optimization_callback([&] {
    ++before_calls;
    return false;
  });
  fixture.refiner->set_after_optimization_callback([&] {
    ++after_calls;
    return true;
  });
  check(!fixture.refine(), "before-hook failure propagates out of RefinePoses");
  check(before_calls == 1 && after_calls == 0 && fixture.angle() == 0.0f
            && fixture.modality->optimization_visualizations == 0,
        "before-hook failure prevents optimization and later stages");

  fixture.modality->reset_counts();
  fixture.refiner->set_before_optimization_callback([&] {
    ++before_calls;
    return fixture.install_prior(fixture.angle());
  });
  fixture.refiner->set_after_optimization_callback([&] {
    ++after_calls;
    return false;
  });
  check(!fixture.refine(), "after-hook failure propagates out of RefinePoses");
  check(before_calls == 2 && after_calls == 1 && fixture.angle() > 0.39f
            && fixture.modality->optimization_visualizations == 0,
        "after-hook failure stops after optimization and before visualization");

  fixture.refiner->set_before_optimization_callback({});
  fixture.refiner->set_after_optimization_callback({});
  fixture.optimizer->ClearJointAnglePriors();
  fixture.joint->ResetJointPoses();
  fixture.modality->reset_counts();
  check(fixture.refine(), "refiner recovers after failed hooks are cleared");
  check(before_calls == 2 && after_calls == 1 && fixture.angle() == 0.0f,
        "cleared failure hooks and priors do not affect later refinement");
}

void prior_moves_only_the_indexed_revolute_joint() {
  OptimizerFixture fixed_base{false};
  check(fixed_base.child_index() == 1,
        "fixed base and parent produce the expected child Jacobian index");
  check(fixed_base.optimizer.SetJointAnglePriors(
            {{fixed_base.child_index(), 0.0f, 0.8f, 0.1f}}),
        "accepts a valid scalar prior");
  check(fixed_base.optimizer.CalculateOptimization(0, 0, 0),
        "optimizes fixed-base fixture");
  check(fixed_base.child_angle() > 0.75f,
        "prior moves the indexed child joint toward its target");
  check(std::abs(fixed_base.parent_angle()) < 1.0e-6f,
        "prior does not move the ancestor joint");
  check(std::abs(fixed_base.base_angle()) < 1.0e-6f,
        "prior does not move a fixed base");

  OptimizerFixture free_base{true};
  check(free_base.child_index() == 2,
        "free base is included before the two revolute child coordinates");
  check(free_base.optimizer.SetJointAnglePriors(
            {{free_base.child_index(), 0.0f, 0.8f, 0.1f}}),
        "accepts a child prior with a free base");
  check(free_base.optimizer.CalculateOptimization(0, 0, 0),
        "optimizes free-base fixture");
  check(free_base.child_angle() > 0.75f,
        "free-base fixture moves the requested child joint");
  check(std::abs(free_base.parent_angle()) < 1.0e-6f,
        "free-base fixture leaves the other revolute joint unchanged");
  check(std::abs(free_base.base_angle()) < 1.0e-6f,
        "free-base fixture leaves the base unchanged");
}

float child_delta_for_standard_deviation(float standard_deviation) {
  OptimizerFixture fixture{false};
  const bool accepted = fixture.optimizer.SetJointAnglePriors(
      {{fixture.child_index(), 0.0f, 1.0f, standard_deviation}});
  check(accepted, "accepts prior used to compare standard deviations");
  check(fixture.optimizer.CalculateOptimization(0, 0, 0),
        "optimizes prior strength fixture");
  return fixture.child_angle();
}

void standard_deviation_changes_factor_strength() {
  const float strong_prior = child_delta_for_standard_deviation(0.1f);
  const float weak_prior = child_delta_for_standard_deviation(1.0f);
  check(strong_prior > weak_prior,
        "smaller standard deviation produces a stronger joint update");
  check(weak_prior > 0.4f,
        "larger standard deviation still contributes a positive update");
}

void zero_residual_prior_adds_hessian_information() {
  OptimizerFixture baseline{false, 0.6f, true};
  check(baseline.optimizer.CalculateOptimization(0, 0, 0),
        "optimizes contradictory soft-constraint baseline");
  const float baseline_change = std::abs(baseline.child_angle() - 0.6f);

  OptimizerFixture with_prior{false, 0.6f, true};
  check(with_prior.optimizer.SetJointAnglePriors(
            {{with_prior.child_index(), 0.6f, 0.6f, 0.1f}}),
        "accepts zero-residual prior");
  check(with_prior.optimizer.CalculateOptimization(0, 0, 0),
        "optimizes with zero-residual prior and contradictory soft constraint");
  const float prior_change = std::abs(with_prior.child_angle() - 0.6f);
  check(baseline_change > 1.0e-3f,
        "soft constraint provides a measurable baseline update");
  check(prior_change < baseline_change,
        "zero-residual prior Hessian resists the contradictory update");
}

void rejects_invalid_priors_without_replacing_existing_values() {
  OptimizerFixture fixture{false};
  const int index = fixture.child_index();
  check(fixture.optimizer.SetJointAnglePriors({{index, 0.0f, 0.25f, 0.1f}}),
        "sets initial prior for atomic validation check");

  check(!fixture.optimizer.SetJointAnglePriors(
            {{index, 0.0f, 0.9f, 0.1f}, {index, 0.0f, 1.0f, 0.1f}}),
        "rejects duplicate Jacobian indices");
  check(!fixture.optimizer.SetJointAnglePriors({{-1, 0.0f, 0.1f, 0.1f}}),
        "rejects negative Jacobian indices");
  check(!fixture.optimizer.SetJointAnglePriors(
            {{2, 0.0f, 0.1f, 0.1f}}),
        "rejects indices outside the fixed-base degrees of freedom");
  check(!fixture.optimizer.SetJointAnglePriors(
            {{index, std::numeric_limits<float>::quiet_NaN(), 0.1f, 0.1f}}),
        "rejects non-finite current angle");
  check(!fixture.optimizer.SetJointAnglePriors(
            {{index, 0.0f, std::numeric_limits<float>::infinity(), 0.1f}}),
        "rejects non-finite target angle");
  check(!fixture.optimizer.SetJointAnglePriors({{index, 0.0f, 0.1f, 0.0f}}),
        "rejects zero standard deviation");
  check(!fixture.optimizer.SetJointAnglePriors(
            {{index, 0.0f, 0.1f,
              std::numeric_limits<float>::infinity()}}),
        "rejects non-finite standard deviation");
  check(!fixture.optimizer.SetJointAnglePriors(
            {{index, 0.0f, 0.1f,
              std::numeric_limits<float>::denorm_min()}}),
        "rejects unrepresentable positive precision");
  check(!fixture.optimizer.SetJointAnglePriors(
            {{index, -std::numeric_limits<float>::max(),
              std::numeric_limits<float>::max(), 1.0f}}),
        "rejects unrepresentable weighted residual");
  check(!fixture.optimizer.SetJointAnglePriors(
            {{index, 0.0f, 0.1f, 0.1f}, {0, 0.0f, 0.1f, 0.1f},
             {1, 0.0f, 0.1f, 0.1f}}),
        "rejects more priors than degrees of freedom");

  check(fixture.optimizer.CalculateOptimization(0, 0, 0),
        "optimizes after rejected prior updates");
  check(std::abs(fixture.child_angle() - 0.25f) < 0.02f,
        "rejected updates leave the prior collection unchanged");
}

void priors_require_optimizer_setup() {
  auto root = std::make_shared<m3t::Link>("unconfigured base");
  m3t::Optimizer optimizer{"unconfigured optimizer", root};
  check(!optimizer.SetJointAnglePriors({}),
        "rejects prior assignment before optimizer setup");
}

void setup_again_clears_priors_when_the_tree_shrinks() {
  OptimizerFixture fixture{false};
  const int removed_child_index = fixture.child_index();
  check(fixture.optimizer.SetJointAnglePriors(
            {{removed_child_index, 0.0f, 0.4f, 0.1f}}),
        "sets a prior for the child that will be removed");

  fixture.parent->ClearChildLinks();
  check(fixture.parent->SetUp(), "sets up parent after removing its child");
  check(fixture.optimizer.SetUp(),
        "sets up optimizer again with fewer degrees of freedom");
  check(fixture.optimizer.CalculateOptimization(0, 0, 0),
        "optimizes after setup clears priors from the old tree");
  check(std::abs(fixture.parent_angle()) < 1.0e-6f,
        "re-setup leaves the remaining joint at its no-prior baseline");
}

void clearing_priors_restores_the_no_prior_update() {
  OptimizerFixture fixture{false};
  check(fixture.optimizer.SetJointAnglePriors(
            {{fixture.child_index(), 0.0f, 0.5f, 0.1f}}),
        "sets prior before clear check");
  check(fixture.optimizer.CalculateOptimization(0, 0, 0),
        "optimizes with prior before clear check");
  check(fixture.child_angle() > 0.4f, "prior changed the child angle");

  fixture.parent->ResetJointPoses();
  fixture.child->ResetJointPoses();
  check(fixture.optimizer.CalculateConsistentPoses(),
        "restores fixture's initial joint poses");
  fixture.optimizer.ClearJointAnglePriors();
  check(fixture.optimizer.CalculateOptimization(0, 0, 0),
        "optimizes after clearing priors");
  check(std::abs(fixture.child_angle()) < 1.0e-6f,
        "clearing priors restores the no-prior baseline");

  check(fixture.optimizer.SetJointAnglePriors(
            {{fixture.child_index(), 0.0f, 0.5f, 0.1f}}),
        "sets prior before empty-vector clear check");
  fixture.parent->ResetJointPoses();
  fixture.child->ResetJointPoses();
  check(fixture.optimizer.CalculateConsistentPoses(),
        "restores initial poses before empty-vector clear check");
  check(fixture.optimizer.SetJointAnglePriors({}),
        "accepts an empty prior collection");
  check(fixture.optimizer.CalculateOptimization(0, 0, 0),
        "optimizes after setting an empty prior collection");
  check(std::abs(fixture.child_angle()) < 1.0e-6f,
        "empty prior collection clears the factor");
}

}  // namespace

int main() {
  visual_weights_scale_the_entire_objective_and_preserve_cad_prior();
  engine_rejects_invalid_visual_link_weights_before_setup();
  fixed_cad_anchor_constrains_only_the_six_dof_base();
  soft_cad_mount_and_encoder_reference_joint_optimize_together();
  native_refiner_refreshes_prior_at_every_update();
  native_refiner_propagates_hook_failures_and_allows_reset();
  prior_moves_only_the_indexed_revolute_joint();
  standard_deviation_changes_factor_strength();
  zero_residual_prior_adds_hessian_information();
  rejects_invalid_priors_without_replacing_existing_values();
  priors_require_optimizer_setup();
  setup_again_clears_priors_when_the_tree_shrinks();
  clearing_priors_restores_the_no_prior_update();
  return failures == 0 ? 0 : 1;
}
