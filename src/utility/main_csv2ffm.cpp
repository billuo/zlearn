#include "base/logger.h"
#include "cli.h"
#include "csv2ffm.h"

using namespace zlearn;

int main(int argc, char* argv[]) {
		logger::initialize();

		CLI::App app("csv2ffm",
					 "utility to convert data from CSV format to FFM format");

		std::string sep;
		app.add_option("--sep", sep, "separator to use instead of comma");

		std::string input, output;
		app.add_option("input", input, "file for csv input")->required();
		app.add_option("output", output, "file for ffm output")->required();

		Flag header = false;
		header.add_to(app, "--header", "--no-header");
		header.on->description("specify that header is present");
		header.off->description("specify that header is absent");

		Flag label = true;
		label.add_to(app, "--label", "--no-label");
		label.on->description("specify that label is present");
		label.off->description("specify that label is absent");

		auto per_col = app.add_flag("--per-column")
						   ->description("process data one column at a time");

		std::string encode;
		app.add_option("--encode", encode)
			->description("specify how features are encoded to real number;\n"
						  "example:\n"
						  "\tgiven that there're 5 features,"
						  "'ccnnc' then treats the first two features"
						  "and the last one as categorical features,"
						  "while the middle two as real features."
						  "To avoid repetition, continuous characters can be "
						  "abbr. into cx, where x is the number of repetition,"
						  " e.g. 'c2n2c'.");
		std::string group;
		app.add_option("--group", group)
			->description("specify how features are grouped into fields;\n"
						  "example:\n"
						  "\tgiven that there're 10 features,"
						  "'1-3,7;4;5-6;8-10' then group features #1,"
						  "#2, #3 and #7 into the same field; features #4"
						  "into a separate field; features #5 and #6 into"
						  "another field and finally features #8 through #10"
						  "into yet another group.");

		CLI11_PARSE(app, argc, argv);

		char separator;
		if (sep.empty()) {
				separator = ',';
		} else if (sep.size() == 1) {
				separator = sep[0];
		} else if (sep == "space") {
				separator = ' ';
		} else if (sep == "tab") {
				separator = '\t';
		} else {
				THROW("unknown separator:'%s'", sep.c_str());
		}

		CSV2FFM reader(separator, label, header);
		reader.per_column = *per_col;
		reader.convert(input, output, encode, group);
}
