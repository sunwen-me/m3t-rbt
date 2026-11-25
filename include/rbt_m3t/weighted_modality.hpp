#pragma once

#include <cmath>
#include <stdexcept>

namespace rbt_m3t {

// Scale the visual objective without changing sampling, noise, or occlusion geometry.
template <typename BaseModality>
class WeightedVisualModality : public BaseModality {
  public:
    using BaseModality::BaseModality;

    void set_visual_weight(double weight) {
        if (!std::isfinite(weight) || weight < 0.01 || weight > 100.0) {
            throw std::invalid_argument("visual link weight must be finite and within [0.01, 100]");
        }
        visual_weight_ = weight;
    }

    bool CalculateGradientAndHessian(int iteration, int corr_iteration,
                                     int opt_iteration) override {
        if (!BaseModality::CalculateGradientAndHessian(iteration, corr_iteration,
                                                       opt_iteration)) {
            return false;
        }
        if (visual_weight_ != 1.0) {
            const auto weight = static_cast<float>(visual_weight_);
            this->gradient_ *= weight;
            this->hessian_ *= weight;
        }
        return true;
    }

  private:
    double visual_weight_ { 1.0 };
};

} // namespace rbt_m3t
