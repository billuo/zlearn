#pragma once

#include "exception.hpp"

#include <filesystem>
#include <fstream>
#include <vector>

NAMESPACE_BEGIN

namespace filesystem = std::filesystem;

std::shared_ptr<std::fstream> must_open_file(filesystem::path file,
											 std::ios_base::openmode mode);
std::string peek_first_line(std::fstream&);

NAMESPACE_END
