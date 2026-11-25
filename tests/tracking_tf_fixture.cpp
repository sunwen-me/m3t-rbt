#include <rbt_m3t/tracking_tf.hpp>
#include <rbt/tf_component.hpp>

#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

struct Options {
    double duration_sec = 10.0, publish_for_sec = 10.0;
    std::string component_id = "m3t-tf-software-fixture";
    rbt::RuntimeOptions runtime;
};

Options parse_options(int argc, char** argv) {
    Options options;
    if (const char* address = std::getenv("RBT_COMPOSER_ADDR"))
        options.runtime.composer_address = address;
    for (int i = 1; i < argc; ++i) {
        const std::string_view key(argv[i]);
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + std::string(key));
        const std::string value(argv[++i]);
        if (key == "--component-id") options.component_id = value;
        else if (key == "--composer") options.runtime.composer_address = value;
        else if (key == "--duration-sec" || key == "--publish-for-sec") {
            double seconds = 0.0;
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), seconds);
            if (error != std::errc() || end != value.data() + value.size()
                || !std::isfinite(seconds) || seconds < 0.0 || seconds > 86400.0
                || (key == "--duration-sec" && seconds < 0.001))
                throw std::invalid_argument("invalid seconds for " + std::string(key));
            (key == "--duration-sec" ? options.duration_sec : options.publish_for_sec) = seconds;
        } else throw std::invalid_argument("unknown option: " + std::string(key));
    }
    if (options.component_id.empty() || options.component_id.size() > 128
        || options.component_id.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") != std::string::npos)
        throw std::invalid_argument("invalid component id");
    if (options.runtime.composer_address.empty()) throw std::invalid_argument("empty composer address");
    return options;
}

struct Fixture {
    rbt_m3t::RobotModel model;
    rbt_m3t::TrackingEstimate estimate;
    rbt_m3t::RectifiedPair pair {};

    Fixture() {
        model.name = "software_fixture";
        model.root_link = "base_link";
        model.links = {{"base_link", {}}, {"gripper_link", {}}};
        rbt_m3t::ModelJoint joint;
        joint.name = "gripper_fixed";
        joint.type = "fixed";
        joint.parent = "base_link";
        joint.child = "gripper_link";
        joint.origin.translation() = Eigen::Vector3f(0.2F, 0.0F, 0.05F);
        joint.origin.linear() = Eigen::AngleAxisf(
            std::numbers::pi_v<float> / 2.0F, Eigen::Vector3f::UnitX()).toRotationMatrix();
        model.joints.push_back(joint);
        estimate.joints_locked_to_feedback = true;
        for (const bool left : {true, false}) {
            const std::string arm = left ? "so101_left" : "so101_right";
            const float sign = left ? 1.0F : -1.0F;
            Eigen::Affine3f root = Eigen::Affine3f::Identity();
            root.translation() = Eigen::Vector3f(sign * 0.1F, sign * 0.2F, left ? 0.8F : 1.0F);
            root.linear() = Eigen::AngleAxisf(sign * std::numbers::pi_v<float> / 2.0F,
                                             Eigen::Vector3f::UnitZ()).toRotationMatrix();
            // Both anchor and correction are nonidentity, while the actual
            // root and analytic camera-to-gripper expectations remain unchanged.
            Eigen::Affine3f correction = Eigen::Affine3f::Identity();
            correction.translation() = Eigen::Vector3f(sign * 0.02F, -0.01F, 0.03F);
            correction.linear() = Eigen::AngleAxisf(sign * 0.15F,
                Eigen::Vector3f::UnitY()).toRotationMatrix();
            const Eigen::Affine3f nominal = root * correction.inverse();
            estimate.mount_refinement.push_back({arm, nominal, root,
                static_cast<double>(correction.translation().norm()),
                static_cast<double>(Eigen::AngleAxisf(correction.rotation()).angle())});
            estimate.links.push_back({arm, "base_link", root});
            estimate.links.push_back({arm, "gripper_link", root * joint.origin});
            estimate.joint_angles[arm] = {{joint.name, 0.0}};
        }
        pair.frame_id = "fixture/color_optical";
        pair.depth_scale = 0.001;
    }
};

struct State {
    Clock::time_point deadline;
    std::atomic<bool> failed {false};
    std::atomic<std::uint64_t> messages {0}, batches {0}, dropped_batches {0}, queue_full_events {0};
};

void expected_poses() {
    // Independent analytic Rz(+/-pi/2)*Rx(pi/2) expectations; quaternions are XYZW.
    std::cout << "SOFTWARE_FIXTURE EXPECTED target=fixture/color_optical "
                 "source=m3t_fixture/so101_left/gripper_link "
                 "translation_m=[0.1,0.4,0.85] rotation_xyzw=[0.5,0.5,0.5,0.5]\n"
                 "SOFTWARE_FIXTURE EXPECTED target=fixture/color_optical "
                 "source=m3t_fixture/so101_right/gripper_link "
                 "translation_m=[-0.1,-0.4,1.05] rotation_xyzw=[0.5,-0.5,-0.5,0.5]\n"
              << std::flush;
}

class FixtureComponent final : public rbt::Component {
  public:
    FixtureComponent(Options options, std::shared_ptr<State> state)
        : options_(std::move(options)), state_(std::move(state)) {}
    rbt::Result<void> prepare() override { return {}; }
    rbt::ComponentId id() const override { return options_.component_id; }
    std::vector<rbt::RequirementSpec> requirements() const override { return {}; }
    rbt::Result<void> activate(rbt::ActivationContext& context) override {
        try {
            auto clock = rbt::time::HostClock::create();
            if (!clock) throw std::runtime_error("HostMonotonic clock unavailable");
            auto publisher = rbt::tf::provide_dynamic_tf<rbt_m3t::TrackingTfCapability>(context);
            std::cout << "SOFTWARE_FIXTURE activated capability="
                      << rbt_m3t::TrackingTfCapability::TYPE_NAME
                      << " authority=" << publisher.capability().binding().descriptor.capability_instance
                      << " hz=30 epoch=0 uncertainty=null\n" << std::flush;
            const auto task = context.spawn_critical("tracking-tf-software-fixture",
                [publisher = std::move(publisher), clock = std::move(clock).value(),
                 state = state_, seconds = options_.publish_for_sec](std::stop_token stop) mutable {
                    try {
                        Fixture fixture;
                        const auto started = Clock::now();
                        auto next = started;
                        const auto period = std::chrono::duration_cast<Clock::duration>(
                            std::chrono::duration<double>(1.0 / 30.0));
                        bool announced_stop = false;
                        std::optional<Clock::time_point> last_pressure_log;
                        while (!stop.stop_requested() && Clock::now() < state->deadline) {
                            const auto tick = Clock::now();
                            if (std::chrono::duration<double>(tick - started).count() < seconds) {
                                auto now = clock.now_monotonic();
                                if (!now) throw std::runtime_error("HostMonotonic timestamp unavailable");
                                fixture.pair.timing.capture = now.value();
                                fixture.pair.timing.received = now.value();
                                fixture.pair.color_capture = now.value();
                                fixture.pair.timing.capture_uncertainty_ns = std::nullopt;
                                const auto records = rbt_m3t::tracking_tf_records(
                                    fixture.model, fixture.estimate, fixture.pair, now.value(), "m3t_fixture");
                                if (records.size() != 6U) throw std::runtime_error("expected six TF records");
                                const auto result = rbt_m3t::publish_tracking_tf_batch(
                                    publisher, records, clock, stop);
                                state->messages += result.published_records;
                                state->queue_full_events += result.queue_full_events;
                                if (result.published_records == records.size()) ++state->batches;
                                else ++state->dropped_batches;
                                if (result.queue_full_events > 0U) {
                                    const auto pressure_time = Clock::now();
                                    if (!last_pressure_log || pressure_time - *last_pressure_log >= 1s) {
                                        std::cerr << "SOFTWARE_FIXTURE queue_full queue_full_events="
                                                  << state->queue_full_events.load()
                                                  << " dropped_batches=" << state->dropped_batches.load()
                                                  << '\n' << std::flush;
                                        last_pressure_log = pressure_time;
                                    }
                                }
                            } else if (!announced_stop) {
                                std::cout << "SOFTWARE_FIXTURE updates_stopped messages=" << state->messages.load()
                                          << " batches=" << state->batches.load()
                                          << " dropped_batches=" << state->dropped_batches.load()
                                          << " queue_full_events=" << state->queue_full_events.load() << '\n' << std::flush;
                                announced_stop = true;
                            }
                            next += period;
                            if (next < Clock::now()) next = Clock::now() + period;
                            std::this_thread::sleep_until(next);
                        }
                    } catch (const std::exception& error) {
                        state->failed = true;
                        std::cerr << "SOFTWARE_FIXTURE publication failed: " << error.what() << '\n' << std::flush;
                        throw;
                    } catch (...) {
                        state->failed = true;
                        std::cerr << "SOFTWARE_FIXTURE publication failed: unknown exception\n" << std::flush;
                        throw;
                    }
                });
            if (!task) throw std::runtime_error("failed to spawn fixture task");
            return {};
        } catch (const std::exception& error) {
            state_->failed = true;
            std::cerr << "SOFTWARE_FIXTURE activation failed: " << error.what() << '\n' << std::flush;
            return rbt::Result<void>::failure(error.what());
        }
    }
  private:
    Options options_;
    std::shared_ptr<State> state_;
};
} // namespace

int main(int argc, char** argv) {
    const auto started = Clock::now();
    try {
        if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
            std::cout << "SOFTWARE_FIXTURE native TF publisher; no cameras or robot hardware\n"
                         "Usage: rbt_m3t_tracking_tf_fixture [--duration-sec N] [--publish-for-sec N]\n"
                         "       [--component-id ID] [--composer HOST:PORT]\n"
                         "Defaults: duration=10s (includes activation wait), publish=10s after activation;\n"
                         "component=m3t-tf-software-fixture; composer uses RBT_COMPOSER_ADDR.\n";
            return 0;
        }
        auto options = parse_options(argc, argv);
        auto state = std::make_shared<State>();
        state->deadline = started + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(options.duration_sec));
        expected_poses();
        rbt::ComponentRuntime runtime(std::make_unique<FixtureComponent>(options, state), options.runtime);
        std::jthread watchdog([&](std::stop_token stop) {
            while (!stop.stop_requested()) {
                if (runtime.state().kind == rbt::ComponentStateKind::Failed) state->failed = true;
                if (state->failed.load() || Clock::now() >= state->deadline) { runtime.stop(); return; }
                std::this_thread::sleep_for(10ms);
            }
        });
        const int result = runtime.run();
        watchdog.request_stop();
        watchdog.join();
        std::cout << "SOFTWARE_FIXTURE stopped messages=" << state->messages.load()
                  << " batches=" << state->batches.load()
                  << " dropped_batches=" << state->dropped_batches.load()
                  << " queue_full_events=" << state->queue_full_events.load()
                  << " failed=" << state->failed.load() << '\n';
        if (options.publish_for_sec > 0.0 && state->messages.load() == 0U) {
            std::cerr << "SOFTWARE_FIXTURE error: publication requested but no TF messages published\n";
            return 2;
        }
        return state->failed.load() ? 1 : result;
    } catch (const std::exception& error) {
        std::cerr << "SOFTWARE_FIXTURE error: " << error.what() << '\n';
        return 2;
    }
}
