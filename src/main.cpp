#include <rbt_m3t/app_config.hpp>
#include <rbt_m3t/joint_prior.hpp>
#include <rbt_m3t/tracking_tf.hpp>
#include <rbt_orbbec/component.hpp>
#include <rbt/runtime.hpp>
#include <rbt/time_stream.hpp>
#include <rbt/tf_component.hpp>

#include <signal.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <syncstream>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {
using namespace std::chrono_literals;
namespace fs = std::filesystem;
using rbt_m3t::RectifiedPair;

// Native contract advertised by the camera-pitch package. Provider identity is
// checked independently; a capability type alone is not a physical authority.
struct PitchTfCapability { static constexpr char TYPE_NAME[] = "camera/PitchTf"; };
using CameraTfClient = rbt::tf::ScopedStaticTfClient<rbt_orbbec::StaticUpdatesCapability,
                                                   rbt_orbbec::StaticSnapshotCapability>;
using PitchTfSubscriber = rbt::FixedSubscriber<rbt::tf::DynamicTransform>;

static_assert(std::atomic<bool>::is_always_lock_free,
              "SIGHUP reload requests require lock-free atomic bool operations");
std::atomic<bool> reload_requested { false };

void request_reload(int) noexcept { reload_requested.store(true, std::memory_order_relaxed); }

class SighupHandlerGuard {
  public:
    explicit SighupHandlerGuard(bool enabled) {
        if (!enabled) return;
        struct sigaction action {};
        action.sa_handler = request_reload;
        action.sa_flags = SA_RESTART;
        if (::sigemptyset(&action.sa_mask) != 0 || ::sigaction(SIGHUP, &action, &previous_) != 0)
            throw std::runtime_error("cannot install SIGHUP handler");
        installed_ = true;
    }
    ~SighupHandlerGuard() {
        if (installed_) (void)::sigaction(SIGHUP, &previous_, nullptr);
    }
    SighupHandlerGuard(const SighupHandlerGuard&) = delete;
    SighupHandlerGuard& operator=(const SighupHandlerGuard&) = delete;
  private:
    struct sigaction previous_ {};
    bool installed_ { false };
};

struct Options {
    bool probe { false }, preview { false }, reload_on_sighup { false }, publish_tf { false };
    std::string provider { "orbbec335" }, component_id { "m3t-so101" };
    fs::path config, snapshot, preview_packet;
    std::uint64_t count { 0U }, duration { 0U }, maximum_skew_ms { 20U };
};

struct Completion {
    std::atomic<bool> done { false };
    std::atomic<int> result { 2 };
    std::atomic<std::uint64_t> frames { 0U };
    std::atomic<bool> snapshot_saved { false };
    void finish(int code) { result.store(code); done.store(true); }
};

constexpr std::size_t kMaximumOutputBytes = 128U * 1024U;
constexpr std::size_t kMaximumPreviewHeaderBytes = 256U * 1024U;

const char* clock_kind_name(rbt::time::ClockKind kind) {
    switch (kind) {
    case rbt::time::ClockKind::HostMonotonic: return "host_monotonic";
    case rbt::time::ClockKind::Device: return "device";
    case rbt::time::ClockKind::SystemRealtime: return "system_realtime";
    }
    return "unknown";
}

template <std::size_t N> std::string text_field(const std::uint8_t (&bytes)[N]) {
    const auto end = std::find(std::begin(bytes), std::end(bytes), 0U);
    if (end == std::end(bytes)) throw std::runtime_error("unterminated camera metadata");
    return {reinterpret_cast<const char*>(bytes), static_cast<std::size_t>(end - std::begin(bytes))};
}

std::uint64_t number(std::string_view text, std::uint64_t maximum) {
    std::uint64_t value = 0U;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc {} || parsed.ptr != text.data() + text.size() || value > maximum)
        throw std::invalid_argument("invalid numeric option: " + std::string(text));
    return value;
}

void usage() {
    std::cout << "rbt-m3t 0.3.0\n"
        "  rbt-m3t probe [--provider orbbec335] [--count 30] [--duration-sec 20]\n"
        "      [--snapshot NEW_DIR] [--max-skew-ms 20] [--preview]\n"
        "  rbt-m3t run --config FILE [--provider orbbec335] [--count N]\n"
        "      [--duration-sec N] [--max-skew-ms 20] [--preview]\n"
        "      [--preview-packet FILE] [--reload-on-sighup] [--publish-tf]\n"
        "--publish-tf publishes <arm>/cad_mount -> <arm>/base_link -> model links via perception/M3tTf.\n"
        "Requires a fixed camera and either encoder-reference links with fixed arm bases\n"
        "or CAD mount refinement. Mount refinement also supports encoder-reference links\n"
        "with free arm bases and soft joint priors; without them, joints stay encoder-locked.\n"
        "With --reload-on-sighup, write the updated measured initialization file first, then\n"
        "send SIGHUP to the running process with its actual PID: kill -HUP <PID>.\n"
        "Requires the camera driver's --align-depth-to-color option.\n";
}

Options arguments(int argc, char** argv) {
    if (argc < 2) throw std::invalid_argument("expected probe or run; see --help");
    Options options;
    const std::string_view command(argv[1]);
    if (command == "probe") { options.probe = true; options.count = 30U; }
    else if (command != "run") throw std::invalid_argument("expected probe or run");
    for (int index = 2; index < argc; ++index) {
        const std::string option(argv[index]);
        if (option == "--publish-tf") { options.publish_tf = true; continue; }
        if (option == "--preview") { options.preview = true; continue; }
        if (option == "--reload-on-sighup") { options.reload_on_sighup = true; continue; }
        if (++index >= argc) throw std::invalid_argument(option + " needs a value");
        const std::string value(argv[index]);
        if (option == "--provider") options.provider = value;
        else if (option == "--component-id") options.component_id = value;
        else if (option == "--config") options.config = fs::absolute(value);
        else if (option == "--snapshot") options.snapshot = fs::absolute(value);
        else if (option == "--preview-packet") {
            if (value.empty()) throw std::invalid_argument("--preview-packet needs a nonempty filename");
            options.preview_packet = fs::absolute(value);
        }
        else if (option == "--count") options.count = number(value, 1'000'000U);
        else if (option == "--duration-sec") options.duration = number(value, 86400U);
        else if (option == "--max-skew-ms") options.maximum_skew_ms = number(value, 100U);
        else throw std::invalid_argument("unknown option: " + option);
    }
    if (options.provider.empty() || options.component_id.empty() || options.maximum_skew_ms == 0U)
        throw std::invalid_argument("provider, component ID and max skew must be nonempty/positive");
    if (options.probe && options.publish_tf)
        throw std::invalid_argument("--publish-tf is only valid for run");
    if (options.probe && options.reload_on_sighup)
        throw std::invalid_argument("--reload-on-sighup is only valid for run");
    if (options.probe && !options.preview_packet.empty())
        throw std::invalid_argument("--preview-packet is only valid for run");
    if (!options.preview_packet.empty()) {
        if (options.preview_packet.filename().empty())
            throw std::invalid_argument("--preview-packet must name a file");
        if (fs::exists(options.preview_packet) && fs::is_directory(options.preview_packet))
            throw std::invalid_argument("--preview-packet must name a file, not a directory");
    }
    if (!options.probe && options.config.empty()) throw std::invalid_argument("run requires --config");
    if (!options.snapshot.empty() && fs::exists(options.snapshot))
        throw std::invalid_argument("snapshot directory must not exist");
    return options;
}

void save_snapshot(const fs::path& directory, const RectifiedPair& pair,
                   const rbt::orbbec::CameraInfoResponse& info) {
    fs::create_directories(directory.parent_path());
    if (!fs::create_directory(directory)) throw std::runtime_error("cannot create new snapshot directory");
    if (!cv::imwrite((directory / "color.png").string(), pair.bgr)
        || !cv::imwrite((directory / "depth.png").string(), pair.depth_u16))
        throw std::runtime_error("cannot write snapshot images");
    std::ofstream file(directory / "calibration.json");
    if (!file) throw std::runtime_error("cannot write snapshot calibration");
    file << std::setprecision(std::numeric_limits<double>::max_digits10)
         << "{\"schema_version\":1,\"rectified\":true,\"width\":" << pair.bgr.cols
         << ",\"height\":" << pair.bgr.rows << ",\"fx\":" << pair.intrinsics.fu
         << ",\"fy\":" << pair.intrinsics.fv << ",\"cx\":" << pair.intrinsics.ppu
         << ",\"cy\":" << pair.intrinsics.ppv << ",\"depth_unit_m\":" << pair.depth_scale
         << ",\"frame_id\":" << rbt_orbbec::json_quote(pair.frame_id)
         << ",\"serial\":" << rbt_orbbec::json_quote(text_field(info.serial))
         << ",\"capture_ns\":" << pair.timing.capture->ns
         << ",\"clock_source\":" << rbt_orbbec::json_quote(pair.timing.capture->clock.source_id().to_hex())
         << ",\"generation\":" << pair.timing.capture->clock.generation()
         << ",\"color_capture_ns\":" << pair.color_capture.ns
         << ",\"received_ns\":" << pair.timing.received.ns
         << ",\"received_clock_kind\":"
         << rbt_orbbec::json_quote(clock_kind_name(pair.timing.received.clock.kind()))
         << ",\"received_clock_source\":"
         << rbt_orbbec::json_quote(pair.timing.received.clock.source_id().to_hex())
         << ",\"received_clock_generation\":" << pair.timing.received.clock.generation()
         << ",\"received_uncertainty_known\":"
         << (pair.timing.received_uncertainty_ns ? "true" : "false")
         << ",\"capture_uncertainty_known\":"
         << (pair.timing.capture_uncertainty_ns ? "true" : "false");
    if (pair.timing.received_uncertainty_ns)
        file << ",\"received_uncertainty_ns\":" << *pair.timing.received_uncertainty_ns;
    if (pair.timing.capture_uncertainty_ns)
        file << ",\"capture_uncertainty_ns\":" << *pair.timing.capture_uncertainty_ns;
    file << ",\"pair_skew_ns\":" << pair.skew_ns << "}\n";
    file.close();
    if (!file) throw std::runtime_error("snapshot calibration write failed");
}

void publish_json(rbt::time::StampedBytesPublisher& publisher, const std::string& json,
                  const RectifiedPair& pair, std::uint64_t sequence, rbt::time::HostClock& host) {
    if (json.size() > kMaximumOutputBytes - rbt::time::kStampedBytesPrefixBytes)
        throw std::runtime_error("tracking result exceeds bounded pose batch");
    auto loan = publisher.loan_slice(rbt::time::kStampedBytesPrefixBytes + json.size());
    if (!loan) throw std::runtime_error("cannot loan tracking pose batch");
    auto sample = std::move(loan).value();
    sample.header() = sequence;
    auto bytes = sample.payload();
    auto now = host.now_monotonic();
    if (!now) throw std::runtime_error("cannot read result receive time");
    auto timing = rbt::time::SensorTiming::create(pair.timing.capture, now.value(),
                                               pair.timing.capture_uncertainty_ns, std::nullopt);
    if (!timing || rbt::time::write_timing_prefix(bytes.first(rbt::time::kStampedBytesPrefixBytes), timing.value())
            != rbt::time::PrefixCodecError::None) throw std::runtime_error("invalid result timing");
    std::memcpy(bytes.data() + rbt::time::kStampedBytesPrefixBytes, json.data(), json.size());
    auto sent = sample.send();
    if (!sent) throw std::runtime_error("cannot publish tracking pose batch");
}

void write_all(int fd, const char* bytes, std::size_t size) {
    while (size != 0U) {
        const auto written = ::write(fd, bytes, size);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            const auto error = written < 0 ? errno : EIO;
            throw std::system_error(error, std::generic_category(), "preview packet write failed");
        }
        const auto count = static_cast<std::size_t>(written);
        bytes += count;
        size -= count;
    }
}

void write_preview_packet(const fs::path& destination, const RectifiedPair& image,
                          const std::string& estimate_json) {
    if (image.bgr.empty() || image.bgr.type() != CV_8UC3)
        throw std::runtime_error("preview packet requires a nonempty BGR8 frame");
    const std::string header = "{\"schema\":\"rbt-m3t-preview-v1\",\"width\":"
        + std::to_string(image.bgr.cols) + ",\"height\":" + std::to_string(image.bgr.rows)
        + ",\"pixel_format\":\"BGR8\",\"estimate\":" + estimate_json + "}\n";
    if (header.size() > kMaximumPreviewHeaderBytes)
        throw std::runtime_error("preview packet header exceeds 256 KiB");

    auto temporary = destination.string() + ".tmp." + std::to_string(::getpid()) + ".XXXXXX";
    std::vector<char> temporary_name(temporary.begin(), temporary.end());
    temporary_name.push_back('\0');
    const int fd = ::mkstemp(temporary_name.data());
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "cannot create preview packet temp file");
    bool open = true;
    try {
        write_all(fd, header.data(), header.size());
        const auto row_bytes = static_cast<std::size_t>(image.bgr.cols) * 3U;
        for (int row = 0; row < image.bgr.rows; ++row)
            write_all(fd, reinterpret_cast<const char*>(image.bgr.ptr(row)), row_bytes);
        if (::close(fd) != 0) {
            open = false;
            throw std::system_error(errno, std::generic_category(), "cannot close preview packet temp file");
        }
        open = false;
        if (::rename(temporary_name.data(), destination.c_str()) != 0)
            throw std::system_error(errno, std::generic_category(), "cannot publish preview packet");
    } catch (...) {
        if (open) (void)::close(fd);
        (void)::unlink(temporary_name.data());
        throw;
    }
}

cv::Mat skeleton_preview(const RectifiedPair& image, const rbt_m3t::TrackingEstimate& estimate,
                         const rbt_m3t::RobotModel& model) {
    auto preview = image.bgr.clone();
    std::map<std::string, std::map<std::string, cv::Point>> points;
    for (const auto& link : estimate.links) {
        const auto p = link.in_camera.translation();
        if (!p.allFinite() || p.z() <= 0.02F) continue;
        const double x = image.intrinsics.fu * p.x() / p.z() + image.intrinsics.ppu;
        const double y = image.intrinsics.fv * p.y() / p.z() + image.intrinsics.ppv;
        if (std::abs(x) > 100000.0 || std::abs(y) > 100000.0) continue;
        points[link.arm][link.link] = {static_cast<int>(std::lround(x)), static_cast<int>(std::lround(y))};
    }
    std::size_t arm_index = 0U;
    for (const auto& [arm, links] : points) {
        const cv::Scalar color = arm_index++ == 0U ? cv::Scalar(50, 230, 50) : cv::Scalar(230, 80, 230);
        for (const auto& joint : model.joints) {
            const auto parent = links.find(joint.parent), child = links.find(joint.child);
            if (parent == links.end() || child == links.end()) continue;
            auto a = parent->second, b = child->second;
            if (cv::clipLine(preview.size(), a, b)) cv::line(preview, a, b, color, 2, cv::LINE_AA);
        }
        for (const auto& [name, point] : links) {
            if (!cv::Rect(0, 0, preview.cols, preview.rows).contains(point)) continue;
            cv::circle(preview, point, 3, color, -1, cv::LINE_AA);
            if (name == model.root_link) cv::putText(preview, arm, point, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }
    }
    return preview;
}

class TrackingComponent final : public rbt::Component {
  public:
    TrackingComponent(Options options, std::shared_ptr<Completion> completion,
                      std::shared_ptr<const rbt_m3t::TrackingConfig> config)
        : options_(std::move(options)), completion_(std::move(completion)), config_(std::move(config)) {}
    rbt::Result<void> prepare() override { return {}; }
    rbt::ComponentId id() const override { return options_.component_id; }
    std::vector<rbt::RequirementSpec> requirements() const override {
        std::vector<rbt::RequirementSpec> result {{"color", rbt_orbbec::ColorCapability::TYPE_NAME},
                {"depth", rbt_orbbec::DepthCapability::TYPE_NAME},
                {"calibration", rbt_orbbec::CameraInfoCapability::TYPE_NAME}};
        if (config_ && config_->secondary_camera) {
            result.push_back({"secondary_color", rbt_orbbec::SecondaryColorCapability::TYPE_NAME});
            result.push_back({"secondary_depth", rbt_orbbec::SecondaryDepthCapability::TYPE_NAME});
            result.push_back({"secondary_calibration",
                              rbt_orbbec::SecondaryCameraInfoCapability::TYPE_NAME});
        }
        if (config_ && config_->camera_motion.enabled) {
            result.push_back({"camera_tf_updates", rbt_orbbec::StaticUpdatesCapability::TYPE_NAME});
            result.push_back({"camera_tf_snapshot", rbt_orbbec::StaticSnapshotCapability::TYPE_NAME});
            result.push_back({"pitch_tf", PitchTfCapability::TYPE_NAME});
        }
        return result;
    }
    rbt::Result<void> activate(rbt::ActivationContext& context) override {
        try {
            const auto color = context.require<rbt_orbbec::ColorCapability>("color");
            const auto depth = context.require<rbt_orbbec::DepthCapability>("depth");
            const auto calibration = context.require<rbt_orbbec::CameraInfoCapability>("calibration");
            if (!color) throw std::runtime_error(color.error());
            if (!depth) throw std::runtime_error(depth.error());
            if (!calibration) throw std::runtime_error(calibration.error());
            for (const auto* binding : {&depth.binding().descriptor, &calibration.binding().descriptor}) {
                const auto& reference = color.binding().descriptor;
                if (binding->provider_component != reference.provider_component
                    || binding->provider_process != reference.provider_process || binding->activation != reference.activation)
                    throw std::runtime_error("RGB-D and calibration must belong to the same camera activation");
            }
            if (color.provider() != options_.provider || depth.provider() != options_.provider
                || calibration.provider() != options_.provider) {
                throw std::runtime_error("camera provider mismatch");
            }
            std::optional<rbt::CapabilityHandle<rbt_orbbec::SecondaryColorCapability>> secondary_color;
            std::optional<rbt::CapabilityHandle<rbt_orbbec::SecondaryDepthCapability>> secondary_depth;
            std::optional<rbt::CapabilityHandle<rbt_orbbec::SecondaryCameraInfoCapability>> secondary_calibration;
            if (config_ && config_->secondary_camera) {
                const auto& expected = *config_->secondary_camera;
                if (expected.provider == options_.provider) {
                    throw std::runtime_error("secondary camera provider must differ from --provider");
                }
                auto sc = context.require<rbt_orbbec::SecondaryColorCapability>("secondary_color");
                auto sd = context.require<rbt_orbbec::SecondaryDepthCapability>("secondary_depth");
                auto si = context.require<rbt_orbbec::SecondaryCameraInfoCapability>("secondary_calibration");
                if (!sc) throw std::runtime_error(sc.error());
                if (!sd) throw std::runtime_error(sd.error());
                if (!si) throw std::runtime_error(si.error());
                const auto& secondary_reference = sc.binding().descriptor;
                for (const auto* binding : {&sd.binding().descriptor, &si.binding().descriptor}) {
                    if (binding->provider_component != secondary_reference.provider_component
                        || binding->provider_process != secondary_reference.provider_process
                        || binding->activation != secondary_reference.activation) {
                        throw std::runtime_error(
                            "secondary RGB-D and calibration must belong to the same camera activation");
                    }
                }
                if (sc.provider() != expected.provider || sd.provider() != expected.provider
                    || si.provider() != expected.provider || sc.provider() == color.provider()) {
                    throw std::runtime_error("secondary camera provider mismatch or duplicate primary provider");
                }
                secondary_color.emplace(std::move(sc));
                secondary_depth.emplace(std::move(sd));
                secondary_calibration.emplace(std::move(si));
            }
            std::shared_ptr<rbt_m3t::CameraMotion> camera_motion;
            std::optional<CameraTfClient> camera_tf;
            std::optional<PitchTfSubscriber> pitch_tf;
            std::string pitch_authority;
            if (config_ && config_->camera_motion.enabled) {
                camera_motion = std::make_shared<rbt_m3t::CameraMotion>(config_->camera_motion);
                const auto updates = context.require<rbt_orbbec::StaticUpdatesCapability>("camera_tf_updates");
                if (!updates) throw std::runtime_error(updates.error());
                const auto& a = color.binding().descriptor;
                const auto& b = updates.binding().descriptor;
                if (a.provider_component != b.provider_component || a.provider_process != b.provider_process
                    || a.activation != b.activation)
                    throw std::runtime_error("RGB-D and SDK TF must belong to the same camera activation");
                const auto pitch = context.require<PitchTfCapability>("pitch_tf");
                if (!pitch || pitch.provider() != config_->camera_motion.pitch_provider)
                    throw std::runtime_error("pitch TF provider mismatch");
                camera_tf.emplace(rbt::tf::require_static_tf<rbt_orbbec::StaticUpdatesCapability,
                                                              rbt_orbbec::StaticSnapshotCapability>(
                    context, "camera_tf_updates", "camera_tf_snapshot", "m3t-camera-sdk",
                    camera_motion->buffer()));
                pitch_authority = pitch.capability_instance();
                const auto expected = rbt::OwnedStreamDescriptor::from(
                    rbt::tf::dynamic_tf_stream_descriptor(pitch_authority));
                const auto* actual = pitch.try_stream();
                if (!actual || actual->service_name != expected.service_name
                    || !actual->message_type.wire_equal(expected.message_type)
                    || actual->profile != expected.profile || actual->transport != expected.transport
                    || actual->resources != expected.resources)
                    throw std::runtime_error("pitch TF endpoint does not match the native TF contract");
                auto subscriber = pitch.subscriber<rbt::tf::DynamicTransform>();
                if (!subscriber) throw std::runtime_error("cannot open native pitch TF stream");
                pitch_tf.emplace(std::move(subscriber).value());
                camera_motion->bind_pitch(pitch_authority);
            }
            auto c = color.slice_subscriber<std::uint64_t, std::uint8_t>();
            auto d = depth.slice_subscriber<std::uint64_t, std::uint8_t>();
            using Rpc = rbt::RpcClient<rbt::orbbec::CameraInfoRequest, rbt::orbbec::CameraInfoResponse>;
            auto descriptor = calibration.rpc().borrowed();
            descriptor.request_type.fields = rbt::FixedMessageBinding<rbt::orbbec::CameraInfoRequest>::descriptor().fields;
            descriptor.response_type.fields = rbt::FixedMessageBinding<rbt::orbbec::CameraInfoResponse>::descriptor().fields;
            auto rpc = Rpc::open_for_activation(descriptor, calibration.capability_instance(),
                                                calibration.binding().descriptor.activation);
            using ImageSubscriber = rbt::SliceSubscriber<std::uint64_t, std::uint8_t>;
            std::optional<ImageSubscriber> secondary_color_subscriber;
            std::optional<ImageSubscriber> secondary_depth_subscriber;
            std::optional<Rpc> secondary_rpc;
            if (secondary_color && secondary_depth && secondary_calibration) {
                auto sc = secondary_color->slice_subscriber<std::uint64_t, std::uint8_t>();
                auto sd = secondary_depth->slice_subscriber<std::uint64_t, std::uint8_t>();
                auto secondary_descriptor = secondary_calibration->rpc().borrowed();
                secondary_descriptor.request_type.fields =
                    rbt::FixedMessageBinding<rbt::orbbec::CameraInfoRequest>::descriptor().fields;
                secondary_descriptor.response_type.fields =
                    rbt::FixedMessageBinding<rbt::orbbec::CameraInfoResponse>::descriptor().fields;
                auto srpc = Rpc::open_for_activation(
                    secondary_descriptor, secondary_calibration->capability_instance(),
                    secondary_calibration->binding().descriptor.activation);
                if (!sc) throw std::runtime_error(std::string("cannot open secondary color stream: ")
                                                  + rbt::to_string(sc.error()));
                if (!sd) throw std::runtime_error(std::string("cannot open secondary depth stream: ")
                                                  + rbt::to_string(sd.error()));
                if (!srpc) throw std::runtime_error(std::string("cannot open secondary CameraInfo RPC: ")
                                                    + rbt::to_string(srpc.error()));
                secondary_color_subscriber.emplace(std::move(sc).value());
                secondary_depth_subscriber.emplace(std::move(sd).value());
                secondary_rpc.emplace(std::move(srpc).value());
            }
            if (!c) throw std::runtime_error(std::string("cannot open color stream: ") + rbt::to_string(c.error()));
            if (!d) throw std::runtime_error(std::string("cannot open depth stream: ") + rbt::to_string(d.error()));
            if (!rpc) throw std::runtime_error(std::string("cannot open CameraInfo RPC: ") + rbt::to_string(rpc.error()));
            std::optional<rbt::time::StampedBytesPublisher> output;
            if (!options_.probe) {
                const auto capability_id = rbt::generate_uuid();
                const auto stream = rbt::StreamDescriptor("rbt/cap/" + capability_id + "/stream",
                    rbt::time::STAMPED_BYTES_DESCRIPTOR, rbt::StreamProfile::Sensor);
                auto publisher = rbt::time::create_stamped_bytes_publisher(stream, kMaximumOutputBytes);
                if (!publisher) throw std::runtime_error("cannot open tracking output stream");
                output.emplace(std::move(publisher).value());
                (void)context.provide<rbt_m3t::TrackingCapability>(capability_id, rbt::OwnedStreamDescriptor::from(stream));
            }
            std::optional<rbt::tf::ScopedDynamicTfPublisher<rbt_m3t::TrackingTfCapability>> tf_output;
            if (options_.publish_tf) {
                tf_output.emplace(rbt::tf::provide_dynamic_tf<rbt_m3t::TrackingTfCapability>(context));
            }
            std::optional<rbt::time::StampedBytesSubscriber> joint_prior_subscriber;
            if (!options_.probe && config_ && config_->joint_prior.enabled) {
                const auto stream = rbt::StreamDescriptor(config_->joint_prior.service_name,
                    rbt::time::STAMPED_BYTES_DESCRIPTOR, rbt::StreamProfile::Sensor);
                auto subscriber = rbt::time::create_stamped_bytes_subscriber(stream);
                if (!subscriber) throw std::runtime_error(std::string("cannot open joint-prior stream: ")
                                                         + rbt::to_string(subscriber.error()));
                joint_prior_subscriber.emplace(std::move(subscriber).value());
            }
            auto options = options_;
            const auto completion = completion_;
            const auto config = config_;
            const auto process = context.process_instance();
            const auto activation = context.activation_id();
            if (pitch_tf) {
                // The native Control stream must keep draining while the GL
                // owner thread builds models or performs expensive tracking.
                (void)context.spawn_critical("m3t-pitch-feedback",
                    [subscriber = std::move(*pitch_tf), camera_motion, pitch_authority]
                    (std::stop_token stop) mutable {
                        try {
                            while (!stop.stop_requested()) {
                                bool received_any = false;
                                for (std::size_t index = 0U; index < 8U; ++index) {
                                    auto received = subscriber.receive();
                                    if (!received) throw std::runtime_error("native pitch TF receive failed");
                                    auto sample = std::move(received).value();
                                    if (!sample) break;
                                    const auto wire = sample->payload();
                                    sample.reset();
                                    camera_motion->ingest_pitch(rbt::tf::decode_dynamic(pitch_authority, wire));
                                    received_any = true;
                                }
                                if (!received_any) std::this_thread::sleep_for(2ms);
                            }
                            return rbt::TaskOutcome::completed();
                        } catch (const std::exception& error) {
                            if (stop.stop_requested()) return rbt::TaskOutcome::completed();
                            return rbt::TaskOutcome::error(error.what());
                        }
                    });
                pitch_tf.reset();
            }
            (void)context.spawn_critical("m3t-rgbd-tracking",
                [options = std::move(options), completion, config, process, activation,
                 color = std::move(c).value(), depth = std::move(d).value(), rpc = std::move(rpc).value(),
                 secondary_color = std::move(secondary_color_subscriber),
                 secondary_depth = std::move(secondary_depth_subscriber),
                 secondary_rpc = std::move(secondary_rpc),
                 tf_output = std::move(tf_output),
                 output = std::move(output), joint_prior_subscriber = std::move(joint_prior_subscriber),
                 camera_motion, camera_tf = std::move(camera_tf)](std::stop_token stop) mutable {
                try {
                    const auto read_info = [](Rpc& client, std::uint32_t stream) {
                        auto response = client.call_copy({stream, 0U}, 1s);
                        if (!response) throw std::runtime_error("camera calibration RPC timed out");
                        return response.value().payload();
                    };
                    const auto color_info = read_info(rpc, 1U);
                    const auto depth_info = read_info(rpc, 2U);
                    rbt_m3t::Rectifier rectifier(color_info, depth_info);
                    if (config && (config->serial != text_field(color_info.serial)
                                   || config->coordinate_frame != text_field(color_info.frame_id)))
                        throw std::runtime_error("initialization does not match camera serial/frame; reinitialize");
                    std::optional<rbt::orbbec::CameraInfoResponse> secondary_color_info;
                    std::optional<rbt::orbbec::CameraInfoResponse> secondary_depth_info;
                    std::unique_ptr<rbt_m3t::Rectifier> secondary_rectifier;
                    std::optional<rbt_m3t::PairBuffer> secondary_pairs;
                    std::optional<rbt_m3t::DualCameraPairBuffer> dual_pairs;
                    std::shared_ptr<rbt_m3t::CameraImages> secondary_images;
                    if (config && config->secondary_camera) {
                        if (!secondary_rpc || !secondary_color || !secondary_depth) {
                            throw std::runtime_error("secondary camera streams are unavailable");
                        }
                        secondary_color_info = read_info(*secondary_rpc, 1U);
                        secondary_depth_info = read_info(*secondary_rpc, 2U);
                        secondary_rectifier = std::make_unique<rbt_m3t::Rectifier>(
                            *secondary_color_info, *secondary_depth_info);
                        const auto& secondary_config = *config->secondary_camera;
                        const auto secondary_serial = text_field(secondary_color_info->serial);
                        const auto secondary_frame = text_field(secondary_color_info->frame_id);
                        if (secondary_serial != secondary_config.serial
                            || secondary_frame != secondary_config.coordinate_frame) {
                            throw std::runtime_error(
                                "secondary camera calibration does not match configured serial/frame");
                        }
                        if (secondary_serial == text_field(color_info.serial)
                            || secondary_frame == text_field(color_info.frame_id)) {
                            throw std::runtime_error(
                                "secondary camera serial and frame must differ from the primary camera");
                        }
                        const auto maximum_skew_ns = secondary_config.maximum_skew_ms * 1'000'000U;
                        secondary_pairs.emplace(options.maximum_skew_ms * 1'000'000U);
                        dual_pairs.emplace(maximum_skew_ns,
                                           secondary_config.time_basis == "received");
                        secondary_images = std::make_shared<rbt_m3t::CameraImages>();
                    }
                    rbt_m3t::PairBuffer pairs(options.maximum_skew_ms * 1'000'000U);
                    auto images = std::make_shared<rbt_m3t::CameraImages>();
                    std::unique_ptr<rbt_m3t::TrackerEngine> engine;
                    std::optional<rbt_m3t::JointPriorBuffer> joint_prior_buffer;
                    if (config && config->joint_prior.enabled) {
                        joint_prior_buffer.emplace(config->joint_prior, config->model, config->arms);
                    }
                    auto host_result = rbt::time::HostClock::create();
                    if (!host_result) throw std::runtime_error("cannot create processing host clock");
                    auto host = std::move(host_result).value();
                    if (camera_tf) camera_tf->refresh_snapshot(1s);
                    auto next_tf_refresh = std::chrono::steady_clock::now() + 2s;
                    const auto poll_camera_tf = [&] {
                        if (!camera_tf) return;
                        (void)camera_tf->drain_updates(8U, 100ms);
                        if (std::chrono::steady_clock::now() >= next_tf_refresh) {
                            camera_tf->refresh_snapshot(100ms);
                            next_tf_refresh = std::chrono::steady_clock::now() + 2s;
                        }
                    };
                    std::uint64_t sequence = 0U;
                    std::uint64_t rejected_joint_prior_samples = 0U;
                    std::uint64_t unavailable_joint_prior_pairs = 0U;
                    std::optional<std::chrono::steady_clock::time_point> last_rejected_sample_diagnostic;
                    std::optional<std::chrono::steady_clock::time_point> last_unavailable_diagnostic;
                    std::optional<std::chrono::steady_clock::time_point> last_camera_tf_diagnostic;
                    bool tf_started = false;
                    std::optional<std::chrono::steady_clock::time_point> last_tf_pressure_log;
                    std::uint64_t unavailable_camera_tf_pairs = 0U;
                    bool camera_motion_started = false;
                    std::size_t joint_prior_poll_calls = 0U;
                    constexpr std::size_t kJointPriorPollLimit = 8U;
                    const auto poll_joint_priors = [&] {
                        if (!joint_prior_subscriber || !joint_prior_buffer) return;
                        while (joint_prior_poll_calls < kJointPriorPollLimit) {
                            ++joint_prior_poll_calls;
                            auto received = joint_prior_subscriber->receive();
                            if (!received) throw std::runtime_error("native joint-prior receive failed");
                            auto sample = std::move(received).value();
                            if (!sample) break;
                            try {
                                joint_prior_buffer->push(rbt_m3t::copy_joint_state_sample(
                                    sample->header(), sample->payload()));
                            } catch (const std::exception& error) {
                                if (rejected_joint_prior_samples != std::numeric_limits<std::uint64_t>::max())
                                    ++rejected_joint_prior_samples;
                                const auto now = std::chrono::steady_clock::now();
                                if (!last_rejected_sample_diagnostic
                                    || now - *last_rejected_sample_diagnostic >= 1s) {
                                    std::string reason(error.what());
                                    if (reason.size() > 160U) reason.resize(160U);
                                    std::osyncstream(std::cerr)
                                        << "{\"event\":\"joint_prior_sample_rejected\",\"count\":"
                                        << rejected_joint_prior_samples << ",\"reason\":"
                                        << rbt_orbbec::json_quote(reason) << "}" << std::endl;
                                    last_rejected_sample_diagnostic = now;
                                }
                            }
                        }
                    };
                    auto last_pair = std::chrono::steady_clock::now();
                    {
                        std::osyncstream activation_event(std::cout);
                        activation_event << "{\"event\":\"rgbd_activation\",\"process_instance\":"
                            << rbt_orbbec::json_quote(process) << ",\"activation\":" << activation
                            << ",\"serial\":" << rbt_orbbec::json_quote(text_field(color_info.serial));
                        if (secondary_color_info) {
                            activation_event << ",\"secondary_serial\":"
                                << rbt_orbbec::json_quote(text_field(secondary_color_info->serial));
                        }
                        activation_event << "}" << std::endl;
                    }
                    while (!stop.stop_requested()) {
                        poll_camera_tf();
                        joint_prior_poll_calls = 0U;
                        poll_joint_priors();
                        bool received_any = false;
                        const auto poll_images = [&](auto& subscriber, rbt_m3t::PairBuffer& buffer) {
                            for (std::size_t index = 0U; index < 4U; ++index) {
                                auto received = subscriber.receive();
                                if (!received) throw std::runtime_error("native RGB-D receive failed");
                                auto sample = std::move(received).value();
                                if (!sample) break;
                                buffer.push(rbt_m3t::copy_image_sample(sample->header(), sample->payload()));
                                received_any = true;
                            }
                        };
                        poll_images(color, pairs);
                        poll_images(depth, pairs);
                        if (secondary_color && secondary_depth && secondary_pairs) {
                            poll_images(*secondary_color, *secondary_pairs);
                            poll_images(*secondary_depth, *secondary_pairs);
                        }

                        using ReadyPair = std::pair<RectifiedPair, std::optional<RectifiedPair>>;
                        std::vector<ReadyPair> ready_pairs;
                        while (auto pair = pairs.take()) {
                            auto rectified = rectifier.apply(*pair);
                            if (dual_pairs) dual_pairs->push(false, std::move(rectified));
                            else ready_pairs.emplace_back(std::move(rectified), std::nullopt);
                        }
                        if (secondary_pairs && secondary_rectifier) {
                            while (auto pair = secondary_pairs->take()) {
                                dual_pairs->push(true, secondary_rectifier->apply(*pair));
                            }
                        }
                        if (dual_pairs) {
                            while (auto matched = dual_pairs->take()) {
                                ready_pairs.emplace_back(std::move((*matched)[0]),
                                                         std::move((*matched)[1]));
                            }
                        }
                        for (auto& ready_pair : ready_pairs) {
                            poll_joint_priors();
                            images->current = std::move(ready_pair.first);
                            if (secondary_images) {
                                if (!ready_pair.second) {
                                    throw std::runtime_error("dual-camera pair omitted its secondary image");
                                }
                                secondary_images->current = std::move(*ready_pair.second);
                            } else if (ready_pair.second) {
                                throw std::runtime_error("single-camera mode received a secondary image");
                            }
                            const auto& image = *images->current;
                            std::optional<rbt_m3t::CameraMotionSelection> camera_selection;
                            if (camera_motion) {
                                // Select the nearest measured feedback explicitly. Only
                                // startup without any feedback needs a bounded wait.
                                const auto deadline = std::chrono::steady_clock::now()
                                    + std::chrono::milliseconds(config->camera_motion.maximum_wait_ms);
                                for (;;) {
                                    try {
                                        poll_camera_tf();
                                        camera_selection = camera_motion->select(image);
                                        images->camera_in_world = camera_selection->camera_in_world;
                                        camera_motion_started = true;
                                        break;
                                    } catch (const rbt::tf::TfError& error) {
                                        if (!camera_motion_started && !stop.stop_requested()
                                            && std::chrono::steady_clock::now() < deadline
                                            && (error.code() == rbt::tf::TfErrorCode::InsufficientHistory
                                                || error.code() == rbt::tf::TfErrorCode::UnknownFrame)) {
                                            std::this_thread::sleep_for(2ms);
                                            continue;
                                        }
                                        ++unavailable_camera_tf_pairs;
                                        const auto now = std::chrono::steady_clock::now();
                                        if (!last_camera_tf_diagnostic || now - *last_camera_tf_diagnostic >= 1s) {
                                            std::osyncstream(std::cerr)
                                                << "{\"event\":\"camera_motion_unavailable\",\"frame_sequence\":"
                                                << sequence << ",\"reason\":"
                                                << rbt_orbbec::json_quote(error.what())
                                                << ",\"time_basis\":"
                                                << rbt_orbbec::json_quote(config->camera_motion.time_basis)
                                                << ",\"skipped_pairs\":" << unavailable_camera_tf_pairs << "}" << std::endl;
                                            last_camera_tf_diagnostic = now;
                                        }
                                        break;
                                    }
                                }
                            }
                            if (secondary_images) {
                                secondary_images->camera_in_world =
                                    images->camera_in_world * config->secondary_camera->camera_in_primary;
                            }
                            const bool camera_unavailable = camera_motion && !camera_selection;
                            rbt_m3t::JointPriorReport prior_report;
                            if (joint_prior_buffer) prior_report = joint_prior_buffer->select(image.timing);
                            const bool prior_unavailable = !options.probe && config
                                && config->joint_prior.enabled && config->joint_prior.required
                                && (prior_report.priors.empty()
                                    || (config->engine.encoder_reference_links
                                        && prior_report.priors.size() != config->joint_prior.mappings.size()));
                            if (prior_unavailable) {
                                if (unavailable_joint_prior_pairs != std::numeric_limits<std::uint64_t>::max())
                                    ++unavailable_joint_prior_pairs;
                                const auto now = std::chrono::steady_clock::now();
                                const bool rate_limit_elapsed = !last_unavailable_diagnostic
                                    || now - *last_unavailable_diagnostic >= 1s;
                                if (rate_limit_elapsed) {
                                    std::osyncstream diagnostic(std::cerr);
                                    diagnostic
                                        << "{\"event\":\"joint_prior_unavailable\",\"frame_sequence\":"
                                        << sequence << ",\"status\":"
                                        << rbt_orbbec::json_quote(prior_report.status)
                                        << ",\"time_basis\":"
                                        << rbt_orbbec::json_quote(prior_report.time_basis)
                                        << ",\"prior_count\":" << prior_report.priors.size()
                                        << ",\"expected_prior_count\":" << config->joint_prior.mappings.size()
                                        << ",\"unavailable_pairs\":" << unavailable_joint_prior_pairs
                                        << ",\"rejected_samples\":" << rejected_joint_prior_samples
                                        << ",\"rejected_mappings\":[";
                                    for (std::size_t index = 0U;
                                         index < prior_report.rejected_mappings.size(); ++index) {
                                        if (index != 0U) diagnostic << ',';
                                        diagnostic << rbt_orbbec::json_quote(
                                            prior_report.rejected_mappings[index]);
                                    }
                                    diagnostic << "]}" << std::endl;
                                    last_unavailable_diagnostic = now;
                                }
                            }
                            std::optional<rbt_m3t::TrackingEstimate> estimate;
                            if (!options.snapshot.empty() && !camera_unavailable && !prior_unavailable) {
                                bool not_saved = false;
                                if (completion->snapshot_saved.compare_exchange_strong(not_saved, true))
                                    save_snapshot(options.snapshot, image, color_info);
                            }
                            if (!options.probe && !prior_unavailable && !camera_unavailable) {
                                if (!engine) {
                                    auto seeds = config->arms;
                                    if (camera_selection)
                                        seeds = camera_motion->camera_seeds(seeds, *camera_selection);
                                    if (config->joint_prior.enabled && config->joint_prior.initialize_from_measurement
                                        && !prior_report.priors.empty()) {
                                        seeds = rbt_m3t::seeds_with_joint_priors(
                                            config->model, seeds, prior_report.priors);
                                    }
                                    engine = std::make_unique<rbt_m3t::TrackerEngine>(
                                        config->model, std::move(seeds), config->engine, images,
                                        secondary_images);
                                    engine->set_up();
                                }
                                if (sequence > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
                                    throw std::runtime_error("M3T iteration limit reached; restart with fresh initialization");
                                if (reload_requested.exchange(false, std::memory_order_relaxed)) {
                                    try {
                                        auto initialization = rbt_m3t::load_tracking_initialization(
                                            config->initialization_path, config->model);
                                        if (initialization.serial != text_field(color_info.serial)
                                            || initialization.coordinate_frame != text_field(color_info.frame_id))
                                            throw std::invalid_argument("initialization does not match camera serial/frame");
                                        if (camera_selection)
                                            initialization.arms = camera_motion->camera_seeds(
                                                initialization.arms, *camera_selection);
                                        engine->reinitialize(initialization.arms, static_cast<int>(sequence),
                                                             prior_report.priors);
                                        std::osyncstream(std::cout)
                                            << "{\"event\":\"tracking_reinitialized\",\"frame_sequence\":"
                                            << sequence << "}" << std::endl;
                                    } catch (const std::invalid_argument& error) {
                                        std::osyncstream(std::cerr)
                                            << "{\"event\":\"reinitialization_failed\",\"frame_sequence\":"
                                            << sequence << ",\"reason\":" << rbt_orbbec::json_quote(error.what())
                                            << "}" << std::endl;
                                    }
                                }
                                estimate = engine->step(static_cast<int>(sequence), prior_report.priors);
                                estimate->joint_prior = prior_report;
                                std::uint64_t dropped_frames = pairs.dropped();
                                const auto add_dropped = [&dropped_frames](std::uint64_t count) {
                                    const auto maximum = std::numeric_limits<std::uint64_t>::max();
                                    dropped_frames = count > maximum - dropped_frames
                                                         ? maximum
                                                         : dropped_frames + count;
                                };
                                if (secondary_pairs) add_dropped(secondary_pairs->dropped());
                                if (dual_pairs) add_dropped(dual_pairs->dropped());
                                const auto* secondary_pair = secondary_images
                                    ? &*secondary_images->current : nullptr;
                                const auto* secondary_config = config && config->secondary_camera
                                    ? &*config->secondary_camera : nullptr;
                                const auto json = rbt_m3t::tracking_json(
                                    *estimate, image, sequence, dropped_frames,
                                    camera_selection ? &*camera_selection : nullptr,
                                    secondary_pair, secondary_config);
                                if (tf_output) {
                                    auto now = host.now_monotonic();
                                    if (!now) throw std::runtime_error("cannot read native TF publish clock");
                                    const auto& nominal_arms = config->tf_nominal_arms.empty()
                                        ? config->arms : config->tf_nominal_arms;
                                    const auto records = rbt_m3t::tracking_tf_records(
                                        config->model, *estimate, image, now.value(), "", nominal_arms);
                                    if (!records.empty() && !tf_started) {
                                        std::osyncstream(std::cout)
                                            << "{\"event\":\"tracking_tf_started\",\"capability\":\"perception/M3tTf\","
                                               "\"frame_prefix\":\"\",\"topology\":\"cad_mount_correction\",\"edges_per_frame\":"
                                            << records.size() << "}" << std::endl;
                                        tf_started = true;
                                    }
                                    const auto sent = rbt_m3t::publish_tracking_tf_batch(
                                        *tf_output, records, host, stop);
                                    const auto pressure_time = std::chrono::steady_clock::now();
                                    if (sent.queue_full_events != 0U
                                        && (!last_tf_pressure_log || pressure_time - *last_tf_pressure_log >= 1s)) {
                                        std::osyncstream(std::cerr)
                                            << "{\"event\":\"tracking_tf_queue_full\",\"frame_sequence\":"
                                            << sequence << ",\"published_edges\":" << sent.published_records
                                            << ",\"expected_edges\":" << records.size() << "}" << std::endl;
                                        last_tf_pressure_log = pressure_time;
                                    }
                                }
                                if (!options.preview_packet.empty())
                                    write_preview_packet(options.preview_packet, image, json);
                                publish_json(*output, json, image, sequence, host);
                                std::osyncstream(std::cout) << json << std::endl;
                            } else if (options.probe) {
                                std::osyncstream(std::cout) << "{\"event\":\"rgbd_pair\",\"sequence\":" << sequence
                                    << ",\"width\":" << image.bgr.cols << ",\"height\":" << image.bgr.rows
                                    << ",\"frame_id\":" << rbt_orbbec::json_quote(image.frame_id)
                                    << ",\"capture_ns\":" << image.timing.capture->ns
                                    << ",\"skew_ns\":" << image.skew_ns
                                    << ",\"depth_unit_m\":" << image.depth_scale
                                    << ",\"dropped_frames\":" << pairs.dropped() << "}" << std::endl;
                            }
                            if (options.preview) {
                                const auto preview = estimate ? skeleton_preview(image, *estimate, config->model) : image.bgr;
                                cv::imshow("RBT M3T (rectified camera frame)", preview);
                                if (cv::waitKey(1) == 27) { completion->finish(0); return rbt::TaskOutcome::completed(); }
                            }
                            ++sequence;
                            if (!prior_unavailable && !camera_unavailable) completion->frames.fetch_add(1U);
                            last_pair = std::chrono::steady_clock::now();
                            if (options.count != 0U && completion->frames.load() >= options.count) {
                                completion->finish(0);
                                return rbt::TaskOutcome::completed();
                            }
                            if (stop.stop_requested()) break;
                        }
                        if (std::chrono::steady_clock::now() - last_pair >= 5s) {
                            throw std::runtime_error(dual_pairs
                                ? "no matched dual-camera RGB-D pair within configured skew limit for five seconds"
                                : "no RGB-D pair within capture-time skew limit for five seconds");
                        }
                        if (!received_any) std::this_thread::sleep_for(2ms);
                    }
                    return rbt::TaskOutcome::completed();
                } catch (const std::exception& error) {
                    if (stop.stop_requested()) return rbt::TaskOutcome::completed();
                    std::osyncstream(std::cerr) << "rbt-m3t: " << error.what() << std::endl;
                    completion->finish(1);
                    return rbt::TaskOutcome::error(error.what());
                }
            });
            return {};
        } catch (const std::exception& error) { return rbt::Result<void>::failure(error.what()); }
    }
  private:
    Options options_;
    std::shared_ptr<Completion> completion_;
    std::shared_ptr<const rbt_m3t::TrackingConfig> config_;
};
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 1 || (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "--version"))) {
            usage(); return 0;
        }
        auto options = arguments(argc, argv);
        std::shared_ptr<const rbt_m3t::TrackingConfig> config;
        if (!options.probe) config = std::make_shared<rbt_m3t::TrackingConfig>(rbt_m3t::load_tracking_config(options.config));
        if (options.publish_tf) {
            if (config->camera_motion.enabled
                || (!config->engine.mount_refinement.enabled && !config->engine.encoder_reference_links)) {
                throw std::invalid_argument("--publish-tf requires a fixed camera and either "
                                            "encoder_reference_links or mount_refinement");
            }
            if (config->engine.encoder_reference_links
                && !config->engine.mount_refinement.enabled
                && !std::ranges::all_of(config->arms, [](const auto& arm) { return arm.fixed_base; })) {
                throw std::invalid_argument("--publish-tf with encoder_reference_links requires fixed arm bases");
            }
        }
        SighupHandlerGuard sighup_handler(options.reload_on_sighup);
        auto completion = std::make_shared<Completion>();
        rbt::RuntimeOptions runtime_options;
        if (const char* address = std::getenv("RBT_COMPOSER_ADDR")) runtime_options.composer_address = address;
        rbt::ComponentRuntime runtime(std::make_unique<TrackingComponent>(options, completion, config), runtime_options);
        std::jthread watchdog([&](std::stop_token stop) {
            const auto start = std::chrono::steady_clock::now();
            while (!stop.stop_requested() && !completion->done.load()) {
                const auto state = runtime.state();
                if (state.kind == rbt::ComponentStateKind::Failed) {
                    std::osyncstream(std::cerr) << "rbt-m3t runtime: " << state.reason << std::endl;
                    completion->finish(1); break;
                }
                if (options.duration != 0U && std::chrono::steady_clock::now() - start >= std::chrono::seconds(options.duration)) {
                    completion->finish(options.count == 0U && completion->frames.load() > 0U ? 0 : 2); break;
                }
                std::this_thread::sleep_for(20ms);
            }
            if (!stop.stop_requested()) runtime.stop();
        });
        const int result = runtime.run();
        watchdog.request_stop();
        return completion->done.load() ? completion->result.load() : result;
    } catch (const std::exception& error) {
        std::cerr << "rbt-m3t: " << error.what() << '\n';
        return 2;
    }
}
