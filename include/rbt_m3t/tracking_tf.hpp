#pragma once

#include <rbt_m3t/model.hpp>

#include <rbt/transforms.hpp>
#include <rbt/tf_component.hpp>

#include <cstddef>
#include <span>
#include <stop_token>
#include <string_view>
#include <vector>

namespace rbt_m3t {

struct TrackingTfCapability {
    inline static constexpr char TYPE_NAME[] = "perception/M3tTf";
};

// Converts a complete estimate into one camera-rooted tree per arm. Every
// tracked link uses canonical arm/link names (an optional fixture prefix may be
// supplied). The tree is camera -> arm/cad_mount (nominal) -> arm/root
// (correction) -> model children. Feedback-locked estimates take their nominal
// poses from the mount diagnostics, whose refined poses must match the roots.
// Encoder-reference estimates take the nominal pose and arm name from each
// nominal_arms entry. The physical roots and links come from the estimate.
// No contact TCP is added. Even constant edges are sampled dynamically, so the
// entire tree disappears from output when capture timing is unavailable or stale.
// Uses the depth capture timestamp and its uncertainty unchanged. Missing,
// unknown, incompatible, future or >500 ms old capture timing returns no records.
// Invalid model trees, frame names or incomplete/non-rigid estimates throw
// std::invalid_argument. Inputs are never modified; no records are published here.
// transform_epoch stays zero within one publisher activation (a restart changes
// authority). Frame sequence remains in JSON and is not a TF transform epoch.
[[nodiscard]] std::vector<rbt::tf::DynamicRecord> tracking_tf_records(
    const RobotModel& model,
    const TrackingEstimate& estimate,
    const RectifiedPair& pair,
    const rbt::time::Timestamp& now,
    std::string_view frame_prefix = "",
    std::span<const ArmSeed> nominal_arms = {});

struct TrackingTfPublishResult {
    std::size_t published_records { 0U };
    std::size_t queue_full_events { 0U };
};

// Native Control subscribers queue four records; a model tree is larger.
// Retry the same edge every 500 us for at most 25 ms per batch under pressure.
// Stop, stale (>500 ms), future or incompatible stamps abandon the remaining
// batch. Original stamps and epochs are never changed. Other errors propagate.
// Per-edge transport is not atomic: consumers still require a complete capture.
[[nodiscard]] TrackingTfPublishResult publish_tracking_tf_batch(
    rbt::tf::ScopedDynamicTfPublisher<TrackingTfCapability>& publisher,
    std::span<const rbt::tf::DynamicRecord> records,
    rbt::time::HostClock& clock,
    std::stop_token stop = {});

} // namespace rbt_m3t
