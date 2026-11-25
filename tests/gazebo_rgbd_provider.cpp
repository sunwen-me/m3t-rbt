// Test-only RBT camera provider for packed Gazebo RGB-D pairs written to stdin.
#include <rbt/orbbec.hpp>
#include <rbt/runtime.hpp>
#include <rbt/time_stream.hpp>
#include <rbt_orbbec/capabilities.hpp>
#include <rbt_orbbec/driver.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <numbers>
#include <sstream>
#include <system_error>
#include <thread>

#include <poll.h>
#include <unistd.h>

namespace {

using namespace std::chrono_literals;
using namespace rbt_orbbec;
using CameraInfoServer = rbt::RpcServer<rbt::orbbec::CameraInfoRequest,
                                        rbt::orbbec::CameraInfoResponse>;

constexpr char kComponentId[] = "gazebo-rgbd";
constexpr char kSerial[] = "GAZEBO_RGBD";
constexpr char kOpticalFrame[] = "gazebo/color_optical";
constexpr std::size_t kTimestampBytes = 8U;
constexpr std::size_t kJointStateBytes = kTimestampBytes + 6U * 8U;
constexpr int kRunning = -1;

struct Camera {
    std::uint32_t width, height, fps;
    double fx, fy, cx, cy;
};

template <typename T>
T number(std::string_view text, const char* field) {
    T result {};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
    if (error != std::errc {} || end != text.data() + text.size()) {
        throw std::invalid_argument(std::string("invalid ") + field);
    }
    return result;
}

Camera parse_camera(char** argv) {
    const Camera camera {number<std::uint32_t>(argv[1], "width"), number<std::uint32_t>(argv[2], "height"),
                         number<std::uint32_t>(argv[3], "fps"), number<double>(argv[4], "fx"),
                         number<double>(argv[5], "fy"), number<double>(argv[6], "cx"),
                         number<double>(argv[7], "cy")};
    if (camera.width == 0U || camera.width > 4096U || camera.height == 0U || camera.height > 4096U
        || camera.fps == 0U) {
        throw std::invalid_argument("camera dimensions must be between 1 and 4096 and FPS must be positive");
    }
    const auto pixels = static_cast<std::size_t>(camera.width) * camera.height;
    if (pixels > (kMaximumFrameBytes - kFrameHeaderBytes) / 3U) {
        throw std::invalid_argument("image dimensions exceed the RGB8 frame size limit");
    }
    const auto fits_float = [](double value) {
        return std::isfinite(value) && std::abs(value) <= std::numeric_limits<float>::max();
    };
    if (!fits_float(camera.fx) || !fits_float(camera.fy) || !fits_float(camera.cx) || !fits_float(camera.cy)
        || static_cast<float>(camera.fx) <= 0.0F || static_cast<float>(camera.fy) <= 0.0F) {
        throw std::invalid_argument("camera intrinsics must fit float and focal lengths must be positive");
    }
    return camera;
}

rbt::orbbec::CameraInfoResponse camera_info(const rbt::orbbec::CameraInfoRequest& request,
                                             const Camera& camera) {
    rbt::orbbec::CameraInfoResponse response {};
    response.stream = request.stream;
    if (request.reserved != 0U || (request.stream != static_cast<std::uint32_t>(Kind::Color)
                                  && request.stream != static_cast<std::uint32_t>(Kind::Depth))) {
        response.status = 2U;
        return response;
    }
    response.width = camera.width;
    response.height = camera.height;
    response.fps = camera.fps;
    response.format = static_cast<std::uint32_t>(request.stream == static_cast<std::uint32_t>(Kind::Color)
                                                      ? Format::Rgb8 : Format::Depth16);
    response.fx = camera.fx;
    response.fy = camera.fy;
    response.cx = camera.cx;
    response.cy = camera.cy;
    std::copy_n(kOpticalFrame, sizeof(kOpticalFrame), response.frame_id);
    std::copy_n(kSerial, sizeof(kSerial), response.serial);
    return response;
}

bool request_ended(rbt::RpcError error) {
    return error == rbt::RpcError::Cancelled || error == rbt::RpcError::DeadlineExceeded;
}

void serve_camera_info(CameraInfoServer& server, const Camera& camera) {
    for (std::size_t served = 0; served < 8U; ++served) {
        auto received = server.try_receive();
        if (!received) throw std::runtime_error("failed receiving CameraInfo request");
        auto request = std::move(received).value();
        if (!request) return;
        auto loan = request->loan_response();
        if (!loan) {
            if (request_ended(loan.error())) continue;
            throw std::runtime_error("failed loaning CameraInfo response");
        }
        auto response = std::move(loan).value();
        response.payload() = camera_info(request->payload(), camera);
        auto sent = response.send();
        if (!sent && !request_ended(sent.error())) {
            throw std::runtime_error("failed sending CameraInfo response");
        }
    }
}

template <typename Capability>
rbt::time::StampedBytesPublisher open_stream(rbt::ActivationContext& context, std::size_t pixel_bytes) {
    const auto capability = rbt::generate_uuid();
    const auto stream = rbt::StreamDescriptor("rbt/cap/" + capability + "/stream",
                                              rbt::time::STAMPED_BYTES_DESCRIPTOR, rbt::StreamProfile::Sensor);
    auto publisher = rbt::time::create_stamped_bytes_publisher(
        stream, rbt::time::kStampedBytesPrefixBytes + kFrameHeaderBytes + pixel_bytes);
    if (!publisher) throw std::runtime_error("failed opening RGB-D stream: " + std::string(rbt::to_string(publisher.error())));
    (void)context.provide<Capability>(capability, stream);
    return std::move(publisher).value();
}

template <typename Encode>
void publish_sample(rbt::time::StampedBytesPublisher& publisher, std::uint64_t sequence,
                    const rbt::time::SensorTiming& timing, std::size_t payload_size, Encode encode) {
    auto loan = publisher.loan_slice(rbt::time::kStampedBytesPrefixBytes + payload_size);
    if (!loan) throw std::runtime_error("failed loaning Gazebo sample: " + std::string(rbt::to_string(loan.error())));
    auto sample = std::move(loan).value();
    sample.header() = sequence;
    auto bytes = sample.payload();
    if (rbt::time::write_timing_prefix(bytes.first(rbt::time::kStampedBytesPrefixBytes), timing)
        != rbt::time::PrefixCodecError::None) {
        throw std::runtime_error("failed encoding Gazebo sample timing");
    }
    encode(bytes.subspan(rbt::time::kStampedBytesPrefixBytes));
    auto sent = sample.send();
    if (!sent) throw std::runtime_error("failed sending Gazebo sample: " + std::string(rbt::to_string(sent.error())));
}

void publish_frame(rbt::time::StampedBytesPublisher& publisher, const Frame& frame,
                   const rbt::time::SensorTiming& timing) {
    publish_sample(publisher, frame.index, timing, kFrameHeaderBytes + frame.data.size(),
        [&](auto bytes) { encode_frame(bytes, frame, kOpticalFrame); });
}

std::uint64_t read_le_u64(std::span<const std::uint8_t, 8> bytes) {
    std::uint64_t value = 0U;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        value |= static_cast<std::uint64_t>(bytes[i]) << (8U * i);
    }
    return value;
}

std::int64_t read_capture_ns(std::span<const std::uint8_t, kTimestampBytes> bytes, std::int64_t previous) {
    const auto value = read_le_u64(bytes);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
        || value <= static_cast<std::uint64_t>(previous)) {
        throw std::invalid_argument("Gazebo frame timestamps must be positive and strictly increasing");
    }
    return static_cast<std::int64_t>(value);
}

struct JointPublisher {
    rbt::time::StampedBytesPublisher publisher;
    std::string session;
    std::uint64_t previous_capture_ns {0U};
    std::uint64_t sequence {0U};
};

void publish_joint_state(JointPublisher& state, std::span<const std::uint8_t, kJointStateBytes> wire,
                         const rbt::time::DeviceClock& clock, std::int64_t image_ns,
                         const rbt::time::Timestamp& received) {
    const auto capture_ns = read_le_u64(wire.first<kTimestampBytes>());
    if (capture_ns == 0U || capture_ns > static_cast<std::uint64_t>(image_ns)
        || capture_ns < state.previous_capture_ns) {
        throw std::invalid_argument("Gazebo joint timestamp must be positive, causal and nondecreasing");
    }

    std::array<double, 6> values {};
    for (std::size_t joint = 0; joint < values.size(); ++joint) {
        const auto value = std::bit_cast<double>(read_le_u64(wire.subspan(kTimestampBytes + 8U * joint).first<8>()));
        values[joint] = joint < 5U ? value * (180.0 / std::numbers::pi) : value;
        if (!std::isfinite(values[joint])) throw std::invalid_argument("Gazebo joint state must be finite");
    }
    if (values.back() < 0.0 || values.back() > 100.0) {
        throw std::invalid_argument("Gazebo jaw percentage must be within [0, 100]");
    }
    // An image can reuse the latest causal joint observation. Keep its original timestamp.
    if (capture_ns == state.previous_capture_ns) return;
    if (state.sequence == 9'007'199'254'740'991U) throw std::runtime_error("Gazebo joint sequence overflow");
    const auto sequence = state.sequence + 1U;
    const auto capture = clock.stamp(static_cast<std::int64_t>(capture_ns));
    if (!capture) throw std::runtime_error("failed stamping Gazebo joint state");
    const auto timing = rbt::time::SensorTiming::create(capture.value(), received, std::nullopt, std::nullopt);
    if (!timing) throw std::runtime_error("invalid Gazebo joint state timing");

    std::ostringstream json;
    json.imbue(std::locale::classic());
    json << std::setprecision(std::numeric_limits<double>::max_digits10)
         << "{\"protocol\":\"rbt.so101.dual.v1\",\"kind\":\"state\","
            "\"units\":\"joint_deg_gripper_percent\",\"sequence\":" << sequence
         << ",\"controller_id\":\"gazebo-so101\",\"driver_session\":" << json_quote(state.session)
         << ",\"fault\":null,\"arms\":{\"left\":[" << values.front();
    for (std::size_t joint = 1U; joint < values.size(); ++joint) json << ',' << values[joint];
    json << "],\"right\":[0,0,0,0,0,0]}}";
    const auto payload = json.str();
    publish_sample(state.publisher, sequence, timing.value(), payload.size(),
        [&](auto bytes) { std::copy(payload.begin(), payload.end(), bytes.begin()); });
    state.sequence = sequence;
    state.previous_capture_ns = capture_ns;
}

class GazeboRgbdProvider final : public rbt::Component {
  public:
    GazeboRgbdProvider(Camera camera, std::string joint_service, std::atomic<int>& result)
        : camera_{camera}, joint_service_{std::move(joint_service)}, result_{result} {}

    rbt::ComponentId id() const override { return kComponentId; }
    rbt::Result<void> prepare() override { return {}; }
    std::vector<rbt::RequirementSpec> requirements() const override { return {}; }

    rbt::Result<void> activate(rbt::ActivationContext& context) override {
        try {
            const auto pixels = static_cast<std::size_t>(camera_.width) * camera_.height;
            auto color = open_stream<ColorCapability>(context, 3U * pixels);
            auto depth = open_stream<DepthCapability>(context, 2U * pixels);
            const auto info_id = rbt::generate_uuid();
            const auto descriptor = rbt::RpcDescriptor::for_fixed<rbt::orbbec::CameraInfoRequest,
                rbt::orbbec::CameraInfoResponse>("rbt/cap/" + info_id + "/rpc");
            auto server = CameraInfoServer::open_or_create_for_capability(descriptor, info_id);
            if (!server) return rbt::Result<void>::failure("failed opening CameraInfo server");
            (void)context.provide_rpc<CameraInfoCapability>(info_id, rbt::OwnedRpcDescriptor::from(descriptor));

            std::optional<JointPublisher> joints;
            if (!joint_service_.empty()) {
                const auto stream = rbt::StreamDescriptor(joint_service_, rbt::time::STAMPED_BYTES_DESCRIPTOR,
                                                          rbt::StreamProfile::Sensor);
                auto publisher = rbt::time::create_stamped_bytes_publisher(
                    stream, rbt::time::kStampedBytesPrefixBytes + 64U * 1024U);
                if (!publisher) return rbt::Result<void>::failure("failed opening Gazebo joint state stream");
                joints.emplace(JointPublisher {std::move(publisher).value(), rbt::generate_uuid()});
            }
            (void)context.spawn_critical("gazebo-rgbd-input",
                [this, pixels, color = std::move(color), depth = std::move(depth),
                 server = std::move(server).value(), joints = std::move(joints)]
                (std::stop_token stop) mutable -> rbt::TaskOutcome {
                    try {
                        stream(stop, pixels, color, depth, server, joints);
                        return rbt::TaskOutcome::completed();
                    } catch (const std::exception& error) {
                        if (stop.stop_requested()) return rbt::TaskOutcome::completed();
                        std::cerr << "rbt_m3t_gazebo_rgbd_provider: " << error.what() << '\n';
                        result_.store(1);
                        return rbt::TaskOutcome::error(error.what());
                    }
                });
            return {};
        } catch (const std::exception& error) {
            return rbt::Result<void>::failure(error.what());
        }
    }

  private:
    void stream(std::stop_token stop, std::size_t pixels, rbt::time::StampedBytesPublisher& color,
                rbt::time::StampedBytesPublisher& depth, CameraInfoServer& server,
                std::optional<JointPublisher>& joints) {
        auto device = rbt::time::DeviceClock::create(parse_uuid(rbt::generate_uuid()));
        auto host = rbt::time::HostClock::create();
        if (!device || !host) throw std::runtime_error("failed creating Gazebo camera clocks");
        std::vector<std::uint8_t> packet(kTimestampBytes + 5U * pixels + (joints ? kJointStateBytes : 0U));
        std::size_t filled = 0;
        std::int64_t previous_ns = 0;
        std::uint64_t index = 0U;
        std::cout << "{\"event\":\"gazebo_rgbd_ready\",\"component_id\":\"" << kComponentId
                  << "\",\"serial\":\"" << kSerial << "\",\"frame_id\":\"" << kOpticalFrame << "\"}" << std::endl;
        while (!stop.stop_requested()) {
            serve_camera_info(server, camera_);
            // Read only after poll, so a stalled writer cannot prevent RPC or shutdown.
            pollfd input {STDIN_FILENO, POLLIN, 0};
            const auto ready = ::poll(&input, 1U, 5);
            if (ready < 0 && errno == EINTR) continue;
            if (ready < 0) throw std::system_error(errno, std::generic_category(), "polling RGB-D stdin");
            if (ready == 0) continue;
            if (input.revents & (POLLERR | POLLNVAL)) throw std::runtime_error("RGB-D stdin is unavailable");
            if (!(input.revents & (POLLIN | POLLHUP))) continue;
            const auto bytes_read = ::read(STDIN_FILENO, packet.data() + filled, packet.size() - filled);
            if (bytes_read < 0 && (errno == EINTR || errno == EAGAIN)) continue;
            if (bytes_read < 0) throw std::system_error(errno, std::generic_category(), "reading RGB-D stdin");
            if (bytes_read == 0) {
                if (filled != 0U) throw std::runtime_error("truncated RGB-D packet at end of stdin");
                result_.store(0);
                return;
            }
            filled += static_cast<std::size_t>(bytes_read);
            if (filled < packet.size()) continue;

            const auto wire = std::span<const std::uint8_t>(packet);
            previous_ns = read_capture_ns(wire.first<kTimestampBytes>(), previous_ns);
            auto capture = device.value().stamp(previous_ns);
            auto received = host.value().now_monotonic();
            if (!capture || !received) throw std::runtime_error("failed stamping Gazebo RGB-D pair");
            auto timing = rbt::time::SensorTiming::create(capture.value(), received.value(), std::nullopt, std::nullopt);
            if (!timing) throw std::runtime_error("invalid Gazebo RGB-D sample timing");
            if (index == std::numeric_limits<std::uint64_t>::max()) throw std::runtime_error("RGB-D frame index overflow");
            ++index;
            if (joints) {
                publish_joint_state(*joints, wire.last<kJointStateBytes>(),
                                    device.value(), previous_ns, received.value());
            }
            const auto rgb = wire.subspan(kTimestampBytes, 3U * pixels);
            const auto millimetres = wire.subspan(kTimestampBytes + 3U * pixels, 2U * pixels);
            publish_frame(color, {Kind::Color, Format::Rgb8, camera_.width, camera_.height,
                                  3U * camera_.width, index, 0U, 1.0, rgb, {}}, timing.value());
            publish_frame(depth, {Kind::Depth, Format::Depth16, camera_.width, camera_.height,
                                  2U * camera_.width, index, 0U, 0.001, millimetres, {}}, timing.value());
            filled = 0;
        }
    }

    Camera camera_;
    std::string joint_service_;
    std::atomic<int>& result_;
};

} // namespace

int main(int argc, char** argv) {
    if (argc != 8 && (argc != 10 || std::string_view(argv[8]) != "--joint-state-service")) {
        std::cerr << "Usage: rbt_m3t_gazebo_rgbd_provider WIDTH HEIGHT FPS FX FY CX CY [--joint-state-service NAME]\n";
        return argc == 2 && std::string_view(argv[1]) == "--help" ? 0 : 2;
    }
    try {
        const auto camera = parse_camera(argv);
        std::string joint_service;
        if (argc == 10) {
            joint_service = argv[9];
            if (joint_service.empty() || joint_service.size() > 240U) {
                throw std::invalid_argument("joint state service name must contain between 1 and 240 bytes");
            }
        }
        std::atomic<int> result {kRunning};
        rbt::RuntimeOptions options;
        if (const char* address = std::getenv("RBT_COMPOSER_ADDR")) options.composer_address = address;
        rbt::ComponentRuntime runtime(
            std::make_unique<GazeboRgbdProvider>(camera, std::move(joint_service), result), options);
        std::jthread watchdog([&](std::stop_token stop) {
            while (!stop.stop_requested() && result.load() == kRunning) {
                const auto state = runtime.state();
                if (state.kind == rbt::ComponentStateKind::Failed) {
                    std::cerr << "rbt_m3t_gazebo_rgbd_provider: " << state.reason << '\n';
                    result.store(1);
                    break;
                }
                std::this_thread::sleep_for(10ms);
            }
            if (!stop.stop_requested()) runtime.stop();
        });
        const auto runtime_result = runtime.run();
        watchdog.request_stop();
        const auto exit_code = result.load();
        return exit_code == kRunning ? runtime_result : exit_code;
    } catch (const std::exception& error) {
        std::cerr << "rbt_m3t_gazebo_rgbd_provider: " << error.what() << '\n';
        return 2;
    }
}
