#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "utils.cuh"

namespace scinthil::microbenchmark {

enum class Benchmark {
  GlobalMemoryRead,
  GlobalMemoryWrite,
  GlobalMemoryReadWrite,
  L1CacheRead,
  L2CacheRead,
  SharedMemoryRead,
  SharedMemoryWrite,
  SharedMemoryReadWrite,
};

struct Arguments {
  Benchmark benchmark{Benchmark::GlobalMemoryRead};
  RunMicrobenchmarkOptions options{};
};

enum class ParseResult {
  Success,
  Help,
  Error,
};

inline void print_usage(const char* program) {
  std::printf(
      "Usage:\n"
      "  %s <global-memory-read|global-memory-write|global-memory-read-write> "
      "[--global-mib-per-buffer N] [--warmup N] [--repeats N] [--device N]\n"
      "  %s <l1-cache-read> "
      "[--l1-kib-per-sm N] [--passes-per-launch N] [--warmup N] [--repeats N] [--device N]\n"
      "  %s <l2-cache-read> "
      "[--l2-working-set-mib N] [--passes-per-launch N] [--warmup N] [--repeats N] [--device N]\n"
      "  %s <shared-memory-read|shared-memory-write|shared-memory-read-write> "
      "[--shared-kib-per-block N] [--passes-per-launch N] [--warmup N] [--repeats N] [--device N]\n",
      program, program, program, program);
}

[[nodiscard]] inline bool parse_number(const char* text, unsigned long long minimum, unsigned long long maximum,
                                       unsigned long long& result) {
  if (text[0] == '-') {
    return false;
  }
  errno = 0;
  char* end{nullptr};
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value < minimum || value > maximum) {
    return false;
  }
  result = value;
  return true;
}

[[nodiscard]] inline bool parse_benchmark(const char* text, Benchmark& benchmark) {
  if (std::strcmp(text, "global-memory-read") == 0) {
    benchmark = Benchmark::GlobalMemoryRead;
  } else if (std::strcmp(text, "global-memory-write") == 0) {
    benchmark = Benchmark::GlobalMemoryWrite;
  } else if (std::strcmp(text, "global-memory-read-write") == 0) {
    benchmark = Benchmark::GlobalMemoryReadWrite;
  } else if (std::strcmp(text, "l1-cache-read") == 0) {
    benchmark = Benchmark::L1CacheRead;
  } else if (std::strcmp(text, "l2-cache-read") == 0) {
    benchmark = Benchmark::L2CacheRead;
  } else if (std::strcmp(text, "shared-memory-read") == 0) {
    benchmark = Benchmark::SharedMemoryRead;
  } else if (std::strcmp(text, "shared-memory-write") == 0) {
    benchmark = Benchmark::SharedMemoryWrite;
  } else if (std::strcmp(text, "shared-memory-read-write") == 0) {
    benchmark = Benchmark::SharedMemoryReadWrite;
  } else {
    return false;
  }
  return true;
}

[[nodiscard]] inline bool is_global_memory_benchmark(Benchmark benchmark) {
  return benchmark == Benchmark::GlobalMemoryRead || benchmark == Benchmark::GlobalMemoryWrite ||
         benchmark == Benchmark::GlobalMemoryReadWrite;
}

[[nodiscard]] inline bool is_shared_memory_benchmark(Benchmark benchmark) {
  return benchmark == Benchmark::SharedMemoryRead || benchmark == Benchmark::SharedMemoryWrite ||
         benchmark == Benchmark::SharedMemoryReadWrite;
}

[[nodiscard]] inline bool is_cache_benchmark(Benchmark benchmark) {
  return benchmark == Benchmark::L1CacheRead || benchmark == Benchmark::L2CacheRead;
}

[[nodiscard]] inline ParseResult parse_arguments(int argc, char** argv, Arguments& arguments) {
  if (argc < 2) {
    std::fprintf(stderr, "missing benchmark name\n");
    return ParseResult::Error;
  }
  if (std::strcmp(argv[1], "--help") == 0) {
    return ParseResult::Help;
  }
  if (!parse_benchmark(argv[1], arguments.benchmark)) {
    std::fprintf(stderr, "unknown benchmark: %s\n", argv[1]);
    return ParseResult::Error;
  }
  for (int index{2}; index < argc; ++index) {
    if (std::strcmp(argv[index], "--help") == 0) {
      return ParseResult::Help;
    }
    if (index + 1 >= argc) {
      std::fprintf(stderr, "missing value for %s\n", argv[index]);
      return ParseResult::Error;
    }

    unsigned long long value{0};
    if (std::strcmp(argv[index], "--global-mib-per-buffer") == 0) {
      if (!is_global_memory_benchmark(arguments.benchmark)) {
        std::fprintf(stderr, "--global-mib-per-buffer is not valid for this benchmark\n");
        return ParseResult::Error;
      }
      if (!parse_number(argv[++index], 1, std::numeric_limits<std::size_t>::max() / Byte2MByte, value)) {
        std::fprintf(stderr, "invalid --global-mib-per-buffer value\n");
        return ParseResult::Error;
      }
      arguments.options.global_mib_per_buffer = static_cast<std::size_t>(value);
    } else if (std::strcmp(argv[index], "--l1-kib-per-sm") == 0) {
      if (arguments.benchmark != Benchmark::L1CacheRead) {
        std::fprintf(stderr, "--l1-kib-per-sm is not valid for this benchmark\n");
        return ParseResult::Error;
      }
      if (!parse_number(argv[++index], SharedMemoryMinimumSizeKib, std::numeric_limits<std::size_t>::max() / Byte2KByte,
                        value)) {
        std::fprintf(stderr, "invalid --l1-kib-per-sm value\n");
        return ParseResult::Error;
      }
      arguments.options.l1_kib_per_sm = static_cast<std::size_t>(value);
    } else if (std::strcmp(argv[index], "--l2-working-set-mib") == 0) {
      if (arguments.benchmark != Benchmark::L2CacheRead) {
        std::fprintf(stderr, "--l2-working-set-mib is not valid for this benchmark\n");
        return ParseResult::Error;
      }
      if (!parse_number(argv[++index], 1, std::numeric_limits<std::size_t>::max() / Byte2MByte, value)) {
        std::fprintf(stderr, "invalid --l2-working-set-mib value\n");
        return ParseResult::Error;
      }
      arguments.options.l2_working_set_mib = static_cast<std::size_t>(value);
    } else if (std::strcmp(argv[index], "--shared-kib-per-block") == 0) {
      if (!is_shared_memory_benchmark(arguments.benchmark)) {
        std::fprintf(stderr, "--shared-kib-per-block is not valid for this benchmark\n");
        return ParseResult::Error;
      }
      if (!parse_number(argv[++index], SharedMemoryMinimumSizeKib, std::numeric_limits<std::size_t>::max() / Byte2KByte,
                        value)) {
        std::fprintf(stderr, "invalid --shared-kib-per-block value\n");
        return ParseResult::Error;
      }
      arguments.options.shared_kib_per_block = static_cast<std::size_t>(value);
    } else if (std::strcmp(argv[index], "--passes-per-launch") == 0) {
      if (!is_shared_memory_benchmark(arguments.benchmark) && !is_cache_benchmark(arguments.benchmark)) {
        std::fprintf(stderr, "--passes-per-launch is not valid for this benchmark\n");
        return ParseResult::Error;
      }
      if (!parse_number(argv[++index], 1, std::numeric_limits<int>::max(), value)) {
        std::fprintf(stderr, "invalid --passes-per-launch value\n");
        return ParseResult::Error;
      }
      arguments.options.passes_per_launch = static_cast<int>(value);
    } else if (std::strcmp(argv[index], "--warmup") == 0) {
      if (!parse_number(argv[++index], 1, std::numeric_limits<int>::max(), value)) {
        std::fprintf(stderr, "invalid --warmup value\n");
        return ParseResult::Error;
      }
      arguments.options.warmup = static_cast<int>(value);
    } else if (std::strcmp(argv[index], "--repeats") == 0) {
      if (!parse_number(argv[++index], 1, std::numeric_limits<int>::max(), value)) {
        std::fprintf(stderr, "invalid --repeats value\n");
        return ParseResult::Error;
      }
      arguments.options.repeats = static_cast<int>(value);
    } else if (std::strcmp(argv[index], "--device") == 0) {
      if (!parse_number(argv[++index], 0, std::numeric_limits<int>::max(), value)) {
        std::fprintf(stderr, "invalid --device value\n");
        return ParseResult::Error;
      }
      arguments.options.device = static_cast<int>(value);
    } else {
      std::fprintf(stderr, "unknown option: %s\n", argv[index]);
      return ParseResult::Error;
    }
  }
  return ParseResult::Success;
}

}  // namespace scinthil::microbenchmark
