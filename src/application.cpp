#include "application_impl.h"
#include "base/enum_db.h"
#include "base/io_util.h"
#include "base/logger.h"
#include "cli.h"
#include "data/sampler.h"
#include "model/metric.h"

NAMESPACE_BEGIN

enum class Algorithm {
		SGD,
		AdaGrad,
		RMSProp,
		Momentum,
		Adam,
		AdamUnbiased,
		AMSGrad,
};

ENUM_DB_DEFINITION(Algorithm) = {
	{Algorithm::SGD, "sgd"},         {Algorithm::AdaGrad, "adagrad"},
	{Algorithm::RMSProp, "rmsprop"}, {Algorithm::Momentum, "momentum"},
	{Algorithm::Adam, "adam"},       {Algorithm::AdamUnbiased, "adam-unbias"},
	{Algorithm::AMSGrad, "amsgrad"},
};

namespace {

CLI::App app(STRINGIFY(PROJECT_NAME), "a better xlearn...?");

CLI::App* help;
std::string help_command;

CLI::App* train;
CLI::App* predict;

std::string input;
std::string output;
std::string summary;
std::string test;
std::string split;
std::string dump;
std::string dump_train;
std::string dump_test;
std::string model;
std::string algorithm;
std::string metric;

CLI::Option* no_output;
Flag regression = true;
Flag normalize = true;
Flag remove_zeros = true;

int n_threads = 0;
real_t learning_rate = 0.001;
real_t lambda_r = 0.0001;
real_t alpha = 0.9;   // for RMSProp
real_t gamma = 0.9;   // for Momentum
real_t beta_1 = 0.9;  // for Adam
real_t beta_2 = 0.99; // for Adam
size_t k = 8;
size_t m = 2;

template <typename E>
std::set<std::string> enum_name_set() {
		std::set<std::string> set;
		EnumDB<E>::get_names(std::inserter(set, set.begin()));
		return set;
}

Optimizer* get_optimizer() {
		switch (to_enum<Algorithm>(algorithm.c_str())) {
		case Algorithm::SGD: return new SGD(learning_rate, lambda_r);
		case Algorithm::AdaGrad: return new AdaGrad(learning_rate, lambda_r);
		case Algorithm::RMSProp:
				return new RMSProp(learning_rate, lambda_r, alpha);
		case Algorithm::Momentum:
				return new Momentum(learning_rate, lambda_r, gamma);
		case Algorithm::Adam:
				return new Adam(learning_rate, lambda_r, beta_1, beta_2);
		case Algorithm::AdamUnbiased:
				return new AdamUnbiased(learning_rate, lambda_r, beta_1,
										beta_2);
		case Algorithm::AMSGrad:
				return new AMSGrad(learning_rate, lambda_r, beta_1, beta_2);
		}
		UNREACHABLE("bad optimization algo: %s", algorithm.c_str());
}

} // namespace

Application::~Application() = default;
Application::Application() : m_impl(new Application_impl) {
		regression.add_to(app, "--regression", "--binary");
		regression.on->description("perform regression");
		regression.off->description("perform binary classification");

		normalize.add_to(app, "--normalize", "--no-normalize");
		normalize.on->description("enable feature vector normalization");
		normalize.off->description("disable feature vector normalization");

		remove_zeros.add_to(app, "--remove-zeros", "--keep-zeros");
		remove_zeros.on->description("remove features of value 0 from input");
		remove_zeros.off->description("keep features of value 0 from input");

		app.add_option("--threads", n_threads)
			->description(
				"number of threads to use for training;\n"
				"positive: use these many threads\n"
				"zero    : use all threads available\n"
				"negative: use these many threads less than available")
			->capture_default_str();

		app.require_subcommand(0, 1);
		{
				help =
					app.add_subcommand("help", "get help about a subcommand");
				help->add_option("name of subcommand", help_command,
								 "name of the subcommand to get help with")
					->required();
		}
		{
				train = app.add_subcommand("train",
										   "train a model either from scratch");

				train->add_option("-i,--input", input)
					->description("path to file to read data from")
					->required();
				auto has_output =
					train->add_option("-o,--output", output)
						->description("path to file to output model to");
				no_output = train->add_flag("--no-output")
								->description("don't save model")
								->excludes(has_output);
				train->add_option("--test", test)
					->description("path to file containing test data");
				train->add_option("--dump", dump)
					->description("path to file to dump model in text");
				train->add_option("--summary", summary)
					->description("path to file to summarize training");

				train->add_option("--split", split)
					->description("specify that input file should be "
								  "randomly split into train-test set");
				train->add_option("--dump-train", dump_train)
					->description("path to file to dump split train data")
					->needs("--split");
				train->add_option("--dump-test", dump_test)
					->description("path to file to dump split test data")
					->needs("--split");

				train->add_set("model", model, {"LM", "FM", "FFM", "HOFM"})
					->description("type of model to train");
				train
					->add_set("--metric", metric, enum_name_set<Metric::Type>())
					->description("metric used to detect early-stopping and "
								  "report to user during training.");
				algorithm = to_string(Algorithm::SGD).c_str();
				train->add_set("--opt", algorithm, enum_name_set<Algorithm>())
					->description("optimization algorithm to use")
					->capture_default_str();

				train->add_option("-n,--epoch", m_impl->n_epochs)
					->description("maximum number of epochs to train")
					->capture_default_str();
				train->add_option("--window", m_impl->window)
					->description("size of early-stop window, i.e. maximum # "
								  "epochs to train while making no progress;"
								  "set to zero to disable early stopping")
					->capture_default_str();
				train->add_option("-r", learning_rate, "learning rate", true);
				train->add_option("--lr", lambda_r, "L2 regularizing", true);
				train->add_option("--alpha", alpha)
					->description("(meaning depends on optimization "
								  "algorithm)")
					->capture_default_str();
				train->add_option("--gamma", gamma)
					->description("(meaning depends on optimization algorithm)")
					->capture_default_str();
				train->add_option("--beta1", beta_1)
					->description("(meaning depends on optimization algorithm)")
					->capture_default_str();
				train->add_option("--beta2", beta_2)
					->description("(meaning depends on optimization algorithm)")
					->capture_default_str();

				train->add_option("-k", k)
					->description("number of latent factors; only has effect "
								  "when training FM and FFM")
					->capture_default_str();
				train->add_option("-m", m)
					->description("order for HOFM")
					->capture_default_str();
		}
		{
				predict = app.add_subcommand("predict",
											 "predict using a existing model");

				predict->add_option("-i,--input", input)
					->description("path to file to load data from")
					->required();

				predict->add_option("-m, --model", model)
					->description("path to file containing model parameters")
					->required();

				predict->add_option("-o,--output", output)
					->description("path to file to output prediction to")
					->required();
		}
}

void Application::parse_options(int argc, char** argv) {
		try {
				app.parse(argc, argv);
		} catch (const CLI::ParseError& e) {
				int status = app.exit(e);
				std::exit(status);
		}
}

int Application::run() {
		if (app.get_subcommands().empty()) {
				printf("%s\n", app.help().c_str());
				return 0;
		}
		if (help->parsed()) {
				if (auto cmd = app.get_subcommand(help_command)) {
						printf("%s\n", cmd->help().c_str());
				} else {
						printf("command %s does not exist\n",
							   help_command.c_str());
				}
				return 0;
		}

		if (n_threads <= 0) {
				n_threads += std::thread::hardware_concurrency();
		}
		m_impl->thread_pool = std::make_shared<ThreadPool>(n_threads);
		logger::info("using {} worker threads", n_threads);

		m_impl->loss = regression ? Loss::Squared : Loss::CrossEntropy;
		if (!metric.empty()) {
				auto mtype = EnumDB<Metric::Type>::to_enum(metric.c_str());
				m_impl->metric = Metric(mtype);
		}

		if (train->parsed()) {
				if (model == "LM")
						m_impl->model = Model::create_LM();
				else if (model == "FM")
						m_impl->model = Model::create_FM(k);
				else if (model == "FFM")
						m_impl->model = Model::create_FFM(k);
				else if (model == "HOFM")
						m_impl->model = Model::create_HOFM(m, k);

				std::shared_ptr<Optimizer> optimizer(get_optimizer());
				ASSERT(optimizer);

				std::shared_ptr<DataSet> train_data, test_data;
				std::shared_ptr<Sampler> train_sampler, test_sampler;

				train_data = DataSet::from_file(input.c_str(), remove_zeros);
				train_data->sort_entries();
				if (!test.empty()) {
						if (!split.empty()) {
								logger::warn(
									"both test file and input split are "
									"specified; test file will be preferred");
						}
						test_data =
							DataSet::from_file(test.c_str(), remove_zeros);
						test_data->sort_entries();
				} else if (!split.empty()) {
						RELEASE_ASSERT(
							std::count(split.begin(), split.end(), ':') == 1);
						auto pos = split.find_first_of(':');
						auto train_share = split.substr(0, pos);
						auto test_share = split.substr(pos + 1);
						logger::info("splitting input into train:test={}:{}",
									 train_share, test_share);
						test_data = train_data->train_test_split(
							std::stoi(train_share), std::stoi(test_share),
							false);
						if (!dump_train.empty()) {
								logger::info("dumping split train data to {}",
											 dump_train.c_str());
								train_data->serialize_txt(dump_train.c_str());
						}
						if (!dump_test.empty()) {
								logger::info("dumping split test data to {}",
											 dump_test.c_str());
								test_data->serialize_txt(dump_test.c_str());
						}
				}
				train_sampler = Sampler::create(train_data);
				train_sampler->set_normalize(normalize);
				test_sampler = Sampler::create(test_data);
				test_sampler->set_normalize(normalize);
				if (!summary.empty())
						m_impl->summary =
							must_open_file(summary, std::ios::out);
				;
				m_impl->train(*optimizer, *train_sampler, test_sampler);

				if (!*no_output) {
						if (output.empty()) output = input + ".bin";
						logger::info("saving model to {}", output.c_str());
						m_impl->model->serialize(output.c_str());
				}

				if (!dump.empty()) {
						logger::info("dumping model to {}", dump.c_str());
						m_impl->model->serialize_txt(dump.c_str());
				}

				return 0;
		} else if (predict->parsed()) {
				m_impl->model = Model::from_file(model.c_str());
				auto data = DataSet::from_file(input.c_str(), remove_zeros);
				data->sort_entries();
				auto sampler = Sampler::create(data);
				sampler->set_normalize(normalize);
				m_impl->predict(*sampler);

				auto f = must_open_file(output.c_str(), std::ios::out);
				for (auto p : m_impl->predicted) {
						*f << p << '\n';
				}
				return 0;
		}
		UNREACHABLE("no subcommand");
}
NAMESPACE_END
