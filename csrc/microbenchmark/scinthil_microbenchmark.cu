#include <cstdlib>

#include "args.cuh"
#include "bandwidth_microbenchmark.cuh"

int main(int argc, char** argv) {
  scinthil::microbenchmark::Arguments arguments{};
  const scinthil::microbenchmark::ParseResult parse_result =
      scinthil::microbenchmark::parse_arguments(argc, argv, &arguments);
  if (parse_result == scinthil::microbenchmark::ParseResult::Help) {
    scinthil::microbenchmark::print_usage(argv[0]);
    return EXIT_SUCCESS;
  }
  if (parse_result == scinthil::microbenchmark::ParseResult::Error) {
    scinthil::microbenchmark::print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  return scinthil::microbenchmark::run_bandwidth_microbenchmark(arguments) ? EXIT_SUCCESS : EXIT_FAILURE;
}
