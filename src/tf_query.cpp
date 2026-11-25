// Native TF consumer. Capture timestamps always come from the received wire.
#include <rbt_m3t/tracking_tf.hpp>
#include <rbt/tf_component.hpp>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <locale>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {
using namespace std::chrono_literals;
using SteadyClock = std::chrono::steady_clock;
using Timestamp = rbt::time::Timestamp;
constexpr std::int64_t kMaxAgeNs = 500'000'000;

struct Options {
    std::string provider;
    std::string parent { "orbbec335/color_optical" };
    std::string child { "so101_left/gripper_link" };
    std::uint64_t count { 1 };
    std::uint64_t timeout_seconds { 10 };
};

std::uint64_t positive(std::string_view text, std::uint64_t maximum) {
    std::uint64_t value {};
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc {} || parsed.ptr != text.data() + text.size()
        || value == 0 || value > maximum)
        throw std::invalid_argument("expected a positive bounded integer: " + std::string(text));
    return value;
}

Options parse_options(int argc, char** argv) {
    // Help must work without initializing transport, even with other arguments.
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--help" || std::string_view(argv[i]) == "-h") {
            std::cout << "Usage: rbt-m3t-tf-query --provider ID [--parent FRAME] [--child FRAME]\n"
                         "       [--count N] [--timeout-sec N]\n"
                         "Defaults: parent=orbbec335/color_optical, "
                         "child=so101_left/gripper_link, count=1, timeout-sec=10.\n"
                         "The m3t target gripper_link is a tracked link, not a contact TCP.\n"
                         "JSON lines retain capture time and clock identity; max age 500 ms.\n"
                         "Unknown timestamp uncertainty is explicitly allowed.\n"
                         "RBT_COMPOSER_ADDR overrides the SDK Composer address.\n";
            std::exit(0);
        }
    }
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (i + 1 >= argc) throw std::invalid_argument(std::string(argument) + " needs a value");
        const std::string_view value(argv[++i]);
        if (argument == "--provider") options.provider = value;
        else if (argument == "--parent") options.parent = value;
        else if (argument == "--child") options.child = value;
        else if (argument == "--count") options.count = positive(value, 1'000'000);
        else if (argument == "--timeout-sec") options.timeout_seconds = positive(value, 86'400);
        else throw std::invalid_argument("unknown option: " + std::string(argument));
    }
    if (options.provider.empty()) throw std::invalid_argument("--provider ID is required");
    (void)rbt::tf::FrameId(options.parent);
    (void)rbt::tf::FrameId(options.child);
    if (options.parent == options.child)
        throw std::invalid_argument("--parent and --child must differ");
    return options;
}

std::string quote(std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result { "\"" };
    for (const unsigned char byte : value) {
        if (byte == '"' || byte == '\\') {
            result += '\\';
            result += static_cast<char>(byte);
        } else if (byte < 0x20) {
            result += "\\u00";
            result += hex[byte >> 4];
            result += hex[byte & 15];
        } else result += static_cast<char>(byte);
    }
    return result + '"';
}

bool same_stamp(const Timestamp& a, const Timestamp& b) {
    return a.clock == b.clock && a.ns == b.ns;
}

bool fresh(const Timestamp& now, const Timestamp& capture) {
    const auto age = rbt::time::checked_difference(now, capture);
    return age && age.value() >= 0 && age.value() <= kMaxAgeNs;
}

struct Completion {
    std::atomic<int> result { -1 }; // pending, success=0, failure=1, timeout=2
    std::mutex mutex;
    std::uint64_t samples {};
    SteadyClock::time_point deadline;

    bool finish(int code) {
        std::lock_guard lock(mutex);
        if (result.load() >= 0) return false;
        result.store(code);
        return true;
    }
};

std::string json_result(const Options& options, const rbt::tf::LookupResult& result) {
    const auto t = result.transform.translation();
    const auto q = result.transform.rotation();
    const auto& stamp = result.requested_timestamp;
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17)
        << "{\"parent\":" << quote(options.parent) << ",\"child\":" << quote(options.child)
        << ",\"provider\":" << quote(options.provider)
        << ",\"target_semantics\":\"tracked_link_not_contact_tcp\""
        << ",\"translation_m\":[" << t.x() << ',' << t.y() << ',' << t.z()
        << "],\"rotation_xyzw\":[" << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w()
        << "],\"capture_ns\":" << stamp.ns
        << ",\"clock\":{\"kind\":\"HostMonotonic\",\"source_id\":"
        << quote(stamp.clock.source_id().to_hex())
        << ",\"generation\":" << stamp.clock.generation() << "},\"uncertainty_ns\":";
    if (result.timing_uncertainty_ns) out << *result.timing_uncertainty_ns;
    else out << "null";
    out << ",\"sources\":[";
    bool first = true;
    for (const auto& source : result.sources) {
        if (!first) out << ',';
        first = false;
        out << "{\"parent\":" << quote(source.parent.str())
            << ",\"child\":" << quote(source.child.str())
            << ",\"authority\":" << quote(source.authority.physical())
            << ",\"transform_epoch\":";
        if (source.transform_epoch) out << *source.transform_epoch;
        else out << "null";
        out << '}';
    }
    return out.str() + "]}";
}

class TfQuery final : public rbt::Component {
  public:
    TfQuery(Options options, std::shared_ptr<Completion> completion)
        : options_(std::move(options)), completion_(std::move(completion)),
          id_("m3t-tf-query-" + rbt::generate_uuid()) {}

    rbt::Result<void> prepare() override { return {}; }
    rbt::ComponentId id() const override { return id_; }
    std::vector<rbt::RequirementSpec> requirements() const override {
        return { { "tracking_tf", rbt_m3t::TrackingTfCapability::TYPE_NAME } };
    }

    rbt::Result<void> activate(rbt::ActivationContext& context) override {
        auto handle = context.require<rbt_m3t::TrackingTfCapability>("tracking_tf");
        if (!handle) return rbt::Result<void>::failure(handle.error());
        if (handle.provider() != options_.provider)
            return rbt::Result<void>::failure("resolved TF provider does not match --provider");
        // Reuse the same descriptor/owner/role validation as require_dynamic_tf.
        const auto expected = rbt::tf::dynamic_tf_stream_descriptor(handle.capability_instance());
        (void)rbt::tf::component_detail::validate_stream_handle(handle, expected);
        auto subscriber = handle.subscriber<rbt::tf::DynamicTransform>();
        if (!subscriber) return rbt::Result<void>(subscriber.error());
        auto clock = rbt::time::HostClock::create();
        if (!clock) return rbt::Result<void>::failure(rbt::time::to_string(clock.error()));
        rbt::tf::LookupPolicy policy;
        policy.allow_unknown_uncertainty = true; // Preserve unknown as null, never zero.
        rbt::tf::TransformBuffer buffer({}, policy);
        auto authority = buffer.bind_provider("tracking_tf", handle.capability_instance());
        const auto options = options_;
        const auto completion = completion_;
        (void)context.spawn_critical("m3t-tf-query",
            [subscriber = std::move(subscriber).value(), buffer = std::move(buffer),
             authority = std::move(authority), clock = std::move(clock).value(), options, completion]
            (std::stop_token stop) mutable {
                try {
                    std::vector<Timestamp> candidates;
                    std::optional<Timestamp> emitted;
                    while (!stop.stop_requested() && completion->result.load() < 0) {
                        // SDK has no latest query. Retain at most four actual capture times.
                        for (std::size_t n = 0; n < rbt::MAX_DRAIN_PER_SOURCE; ++n) {
                            auto received = subscriber.receive();
                            if (!received) throw std::runtime_error("native TF receive failed");
                            auto sample = std::move(received).value();
                            if (!sample) break;
                            const auto wire = sample->payload();
                            sample.reset(); // Release the native loan before ingest/lookup.
                            auto record = rbt::tf::decode_dynamic(authority.physical(), wire);
                            const auto stamp = record.stamp;
                            buffer.ingest_dynamic(authority, std::move(record));
                            if (stamp.clock != clock.monotonic_identity()) continue;
                            if (std::none_of(candidates.begin(), candidates.end(),
                                    [&](const auto& candidate) { return same_stamp(candidate, stamp); })) {
                                candidates.push_back(stamp);
                                std::sort(candidates.begin(), candidates.end(),
                                    [](const auto& a, const auto& b) { return a.ns > b.ns; });
                                if (candidates.size() > 4) candidates.pop_back();
                            }
                        }
                        for (const auto& stamp : candidates) {
                            if (emitted && stamp.ns <= emitted->ns) continue;
                            auto now = clock.now_monotonic();
                            if (!now) throw std::runtime_error("HostClock read failed");
                            if (!fresh(now.value(), stamp)) continue;
                            try {
                                const auto result = buffer.lookup_at(options.parent, options.child, stamp);
                                // No interpolation or mixing paths from different captures.
                                if (result.sources.empty()
                                    || !std::all_of(result.sources.begin(), result.sources.end(),
                                        [&](const auto& source) {
                                            return source.authority == authority
                                                && same_stamp(source.bracket_start, stamp)
                                                && same_stamp(source.bracket_end, stamp);
                                        })) continue;
                                now = clock.now_monotonic();
                                if (!now || !fresh(now.value(), stamp)) continue;
                                const auto json = json_result(options, result);
                                std::lock_guard lock(completion->mutex);
                                if (completion->result.load() >= 0) return rbt::TaskOutcome::completed();
                                if (SteadyClock::now() >= completion->deadline) break;
                                std::cout << json << std::endl;
                                if (!std::cout) throw std::runtime_error("JSON output failed");
                                emitted = stamp;
                                if (++completion->samples >= options.count) {
                                    completion->result.store(0);
                                    return rbt::TaskOutcome::completed();
                                }
                                break;
                            } catch (const rbt::tf::TfError&) {
                                // Missing/partial/incompatible paths produce no output; retry.
                            }
                        }
                        // The publisher sends a full two-arm tree through a
                        // four-record queue and waits at most 25 ms per batch.
                        // Drain often enough for concurrent query consumers.
                        std::this_thread::sleep_for(1ms);
                    }
                    return rbt::TaskOutcome::completed();
                } catch (const std::exception& error) {
                    if (stop.stop_requested()) return rbt::TaskOutcome::completed();
                    completion->finish(1);
                    std::cerr << "rbt-m3t-tf-query: " << error.what() << '\n';
                    return rbt::TaskOutcome::error(error.what());
                }
            });
        return {};
    }

  private:
    Options options_;
    std::shared_ptr<Completion> completion_;
    rbt::ComponentId id_;
};
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        auto completion = std::make_shared<Completion>();
        completion->deadline = SteadyClock::now() + std::chrono::seconds(options.timeout_seconds);
        rbt::RuntimeOptions runtime_options;
        if (const char* address = std::getenv("RBT_COMPOSER_ADDR"))
            runtime_options.composer_address = address;
        rbt::ComponentRuntime runtime(std::make_unique<TfQuery>(options, completion), runtime_options);
        std::jthread watchdog([&](std::stop_token stop) {
            while (!stop.stop_requested() && completion->result.load() < 0) {
                if (runtime.state().kind == rbt::ComponentStateKind::Failed) {
                    completion->finish(1);
                    break;
                }
                if (SteadyClock::now() >= completion->deadline) {
                    if (completion->finish(2))
                        std::cerr << "rbt-m3t-tf-query: timed out waiting for fresh complete TF captures\n";
                    break;
                }
                std::this_thread::sleep_for(10ms);
            }
            if (!stop.stop_requested()) runtime.stop();
        });
        (void)runtime.run();
        watchdog.request_stop();
        watchdog.join();
        const int result = completion->result.load();
        return result < 0 ? 1 : result;
    } catch (const std::exception& error) {
        std::cerr << "rbt-m3t-tf-query: " << error.what() << '\n';
        return 2;
    }
}
