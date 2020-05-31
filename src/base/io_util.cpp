#include "io_util.h"

#include "logger.h"

NAMESPACE_BEGIN

std::shared_ptr<std::fstream> must_open_file(filesystem::path file,
											 std::ios_base::openmode mode) {
		if (not filesystem::exists(file)) {
				if (mode & std::ios::in)
						throw Exception(
							"failed to open '%s': file does not exist",
							file.c_str());
				if (mode & std::ios::out)
						logger::info("will create '{}' for writing",
									 file.c_str());
		} else if (mode & std::ios::out)
				logger::info("will overwrite '{}'", file.c_str());

		auto ret = std::make_shared<std::fstream>(file, mode);
		if (!ret->is_open())
				throw Exception("failed to open '%s'", file.c_str());
		return ret;
}

std::string peek_first_line(std::fstream& ifs) {
		std::string str;
		std::getline(ifs, str).seekg(0, std::ios::beg);
		return str;
}

NAMESPACE_END
