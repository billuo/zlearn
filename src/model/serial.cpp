#include "base/serial.h"
#include "base/io_util.h"
#include "ffm.h"
#include "fm.h"
#include "hofm.h"
#include "lm.h"

NAMESPACE_BEGIN

std::unique_ptr<Model> Model::from_file(const String& filename) {
		std::unique_ptr<Model> ret;
		auto file =
			must_open_file(filename.c_str(), std::ios::in | std::ios::binary);
		auto model = peek_first_line(*file);
		if (model == "LM") {
				ret.reset(new LM());
		} else if (model == "FM") {
				ret.reset(new FM(0));
		} else if (model == "FFM") {
				ret.reset(new FFM(0));
		} else if (model == "HOFM") {
				ret.reset(new HOFM(0, 0));
		} else {
				THROW("unknown model file(%s): %s", filename.c_str(),
					  model.c_str());
		}
		ret->deserialize(filename);
		return ret;
}

void LM::serialize_txt(const String& filename) const {
		auto file = must_open_file(filename.c_str(), std::ios::out);
		*file << "bias: " << b[0] << std::endl;
		for (index_t i = 0; i < w.rows(); ++i) {
				*file << "w_" << i << ": " << w.row(i)[0] << std::endl;
		}
}

void LM::serialize(const String& filename) const {
		auto file =
			must_open_file(filename.c_str(), std::ios::out | std::ios::binary);
		*file << "LM\n";
		serialize_matrix(*file, b.col(0));
		serialize_matrix(*file, w.col(0));
}

void LM::deserialize(const String& filename) {
		auto file =
			must_open_file(filename.c_str(), std::ios::in | std::ios::binary);
		String model;
		*file >> model;
		ASSERT(model == "LM");
		deserialize_matrix(*file, b);
		deserialize_matrix(*file, w);
		n = w.rows();
}

void FM::serialize_txt(const String& filename) const {
		auto file = must_open_file(filename.c_str(), std::ios::out);
		*file << "bias: " << b[0] << std::endl;
		for (index_t i = 0; i < w.rows(); ++i) {
				*file << "w_" << i << ": " << w.row(i)[0] << std::endl;
		}
		for (index_t i = 0; i < v.rows(); ++i) {
				*file << "v_" << i << ": " << v.row(i).head(k) << std::endl;
		}
}

void FM::serialize(const String& filename) const {
		auto file =
			must_open_file(filename.c_str(), std::ios::out | std::ios::binary);
		*file << "FM\n";
		serialize_matrix(*file, b.col(0));
		serialize_matrix(*file, w.col(0));
		serialize_matrix(*file, v.leftCols(k));
}

void FM::deserialize(const String& filename) {
		auto file =
			must_open_file(filename.c_str(), std::ios::in | std::ios::binary);
		String model;
		*file >> model;
		ASSERT(model == "FM");
		deserialize_matrix(*file, b);
		deserialize_matrix(*file, w);
		deserialize_matrix(*file, v);
		n = w.rows();
		k = v.cols();
}
void FFM::serialize_txt(const String& filename) const {
		auto file = must_open_file(filename.c_str(), std::ios::out);
		*file << "bias: " << b[0] << std::endl;
		for (index_t i = 0; i < w.rows(); ++i) {
				*file << "w_" << i << ": " << w.row(i)[0] << std::endl;
		}
		for (index_t i = 0; i < v.rows(); ++i) {
				for (size_t j = 0; j < n_f; ++j) {
						*file << "v_" << i << "_f" << j << ": " << get_v(i, j)
							  << std::endl;
				}
		}
}

void FFM::serialize(const String& filename) const {
		auto file =
			must_open_file(filename.c_str(), std::ios::out | std::ios::binary);
		*file << "FFM\n";
		serialize_matrix(*file, b.col(0));
		serialize_matrix(*file, w.col(0));
		serialize_to(*file, n_f);

		// TODO: hacky
		serialize_to(*file, n);
		serialize_to(*file, n_f * k);
		for (index_t r = 0; r < n; ++r) {
				for (size_t f = 0; f < n_f; ++f) {
						index_t base = f * k * (1 + m_extras.size());
						for (index_t c = base; c < base + k; ++c) {
								serialize_to(*file, v.coeff(r, c));
						}
				}
		}
}
void FFM::deserialize(const String& filename) {
		auto file =
			must_open_file(filename.c_str(), std::ios::in | std::ios::binary);
		String model;
		*file >> model;
		ASSERT(model == "FFM");
		deserialize_matrix(*file, b);
		deserialize_matrix(*file, w);
		deserialize_from(*file, n_f);
		for (size_t f = 0; f < n_f; ++f) {}
		deserialize_matrix(*file, v);
		n = w.rows();
		k = v.cols() / n_f;
}

void HOFM::serialize_txt(const String& filename) const {
		auto file = must_open_file(filename.c_str(), std::ios::out);
		*file << "bias: " << b[0] << std::endl;
		for (index_t i = 0; i < w.rows(); ++i) {
				*file << "w_" << i << ": " << w.row(i)[0] << std::endl;
		}
		for (size_t t = 0; t < v.size(); ++t) {
				auto& p = v[t];
				for (index_t i = 0; i < p.rows(); ++i) {
						*file << 'd' << t << "v_" << i << ": "
							  << p.row(i).head(k) << std::endl;
				}
		}
}

void HOFM::serialize(const String& filename) const {
		auto file =
			must_open_file(filename.c_str(), std::ios::out | std::ios::binary);
		*file << "HOFM\n";
		serialize_matrix(*file, b.col(0));
		serialize_matrix(*file, w.col(0));
		serialize_to(*file, v.size());
		for (auto& p : v) {
				serialize_matrix(*file, p.leftCols(k));
		}
}

void HOFM::deserialize(const String& filename) {
		auto file =
			must_open_file(filename.c_str(), std::ios::in | std::ios::binary);
		String model;
		*file >> model;
		ASSERT(model == "HOFM");
		deserialize_matrix(*file, b);
		deserialize_matrix(*file, w);
		size_t sz;
		deserialize_from(*file, sz);
		order = sz + 1;
		v.resize(sz);
		for (auto& p : v) {
				deserialize_matrix(*file, p);
		}
		n = w.rows();
		k = v[0].cols();
}

NAMESPACE_END
