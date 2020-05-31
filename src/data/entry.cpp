#include "entry.h"

#include <algorithm>

NAMESPACE_BEGIN

void Entry::set_normalize(bool b) {
		if (b) {
				if (inv_norm2) return; // already normalized
				// TODO normalized but feature value was changed? allow update

				real_t norm2 = 0.0;
				for (auto& f : features) {
						norm2 += f.value * f.value;
				}
				inv_norm2 = 1.0 / norm2;
				auto inv_norm = std::sqrt(*inv_norm2);
				for (auto& f : features) {
						f.normalized_value = f.value * inv_norm;
				}
		} else {
				if (!inv_norm2) return; // not normalized in the beginning

				inv_norm2.reset();
				for (auto& f : features) {
						f.normalized_value = f.value;
				}
		}
}
void Entry::sort_features() {
		std::sort(
			features.begin(), features.end(),
			[](const Feature& f1, const Feature& f2) { return f1.id < f2.id; });
}

NAMESPACE_END
