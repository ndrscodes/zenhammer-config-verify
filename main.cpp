#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <emmintrin.h>
#include <immintrin.h>
#include <random>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <x86intrin.h>
#include <sched.h>
#include "DRAMAddr.hpp"
#include "DRAMConfig.hpp"

const size_t N_PAGES = 1;
const size_t N_SAMPLES_PER_AVG = 128;
const size_t N_AVGS_PER_MEASUREMENT = 17;
const size_t THRESHOLD_SAMPLES = 1024 * 5;
const size_t N_BUCKETS = 200;
const size_t THRESHOLD_WINDOW_SIZE = 5;
const size_t PAGE_2MB = 1024 * 1024 * 2;
const size_t PAGE_1GB = 1024 * 1024 * 1024;
static size_t PAGE_SIZE = PAGE_1GB;

typedef struct {
  double_t lower_bound;
  double_t upper_bound;
} area;

typedef struct {
  size_t start;
  size_t end;
  bool high_activity;
} activity;

typedef struct {
  area ar;
  uint32_t count;
} bucket;

bool verbose = false;

void log_v(bool force, std::string text, va_list args) {
  if(!verbose && !force) {
    return;
  }
  if(text.compare(text.length() - 1, 1, "\n") != 0) {
    text.append("\n");
  }

  vprintf(text.c_str(), args);
}

void log_err(std::string text, ...) {
  va_list args;
  va_start(args, text);
  log_v(true, text, args);
}

void log(std::string text, ...) {
  va_list args;
  va_start(args, text);
  log_v(false, text, args);
}

void* allocate(size_t n_pages) {
  uint64_t huge_map = 0;
  if(PAGE_SIZE == PAGE_1GB) {
    huge_map = MAP_HUGE_1GB;
  } else if(PAGE_SIZE == PAGE_2MB) {
    huge_map = MAP_HUGE_2MB;
  }
  void* res = mmap(NULL, n_pages * PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE | MAP_HUGETLB | huge_map | MAP_POPULATE, -1, 0);
  if(res == MAP_FAILED) {
    log("memory allocation failed (%s)\n", strerror(errno));
    exit(EXIT_FAILURE);
  }
  memset(res, 0x42, n_pages * PAGE_SIZE);
  log("allocated %lu bytes of memory.\n", n_pages * PAGE_SIZE);
  return res;
}

uint64_t median(std::vector<uint64_t> &vec) {
  size_t n = vec.size() / 2;
  std::nth_element(vec.begin(), vec.begin() + n, vec.end());
  return vec[n];
}

uint32_t sum(bucket buckets[], size_t n_buckets) {
  uint32_t sum = 0;
  for(size_t i = 0; i < n_buckets; i++) {
    sum += buckets[i].count;
  }
  return sum;
}

double_t avg(bucket buckets[], size_t n_buckets) {
  return sum(buckets, n_buckets) / (double_t)n_buckets;
}

void* random_page_addr(void *start_addr, size_t alloc_size, size_t step_size) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<> distr(0, (alloc_size * PAGE_SIZE / step_size) - 1);

  size_t random = distr(gen) * step_size;

  return ((char *)start_addr) + random;
}

uint64_t measure_timing(volatile char *addr1, volatile char *addr2) {
  unsigned int tsc_aux;
  std::vector<uint64_t> avgs;
  avgs.reserve(N_AVGS_PER_MEASUREMENT);

  log("measuring timing between %s and %s\n", DRAMAddr((void *)addr1).to_string().c_str(), DRAMAddr((void *)addr2).to_string().c_str());

  for(int i = 0; i < N_AVGS_PER_MEASUREMENT; i++) {
    sched_yield();
    uint64_t sum = 0;

    for(int j = 0; j < N_SAMPLES_PER_AVG; j++) {
      _mm_clflushopt((void *)addr1);
      _mm_clflushopt((void *)addr2);
      _mm_mfence();

      uint64_t start = __rdtscp(&tsc_aux);
      _mm_mfence();
      *addr1;
      *addr2;
      _mm_mfence();
      uint64_t end = __rdtscp(&tsc_aux);

      sum += end - start;
    }
    
    avgs.push_back(sum / N_SAMPLES_PER_AVG);
  }
  
  return median(avgs);
}

bucket* to_hist(std::vector<uint64_t> access_times, size_t n_buckets) {
  bucket* hist = new bucket[n_buckets];
  std::sort(access_times.begin(), access_times.end());
  uint64_t min = access_times.front();
  uint64_t max = access_times.back();
  double_t window_size = ((double_t)(max - min)) / n_buckets;
  size_t current_bucket = 0;
  double_t current_window = min + window_size;
  hist[0].ar = {current_window - window_size, current_window};

  for(size_t i = 0; i < access_times.size(); i++) {
    while(access_times[i] > current_window && current_bucket + 1 < n_buckets) {
      current_bucket++;
      current_window += window_size;
      hist[current_bucket].ar = {current_window - window_size, current_window};
    }
    hist[current_bucket].count = hist[current_bucket].count + 1;
  }

  return hist;
}

std::vector<activity> find_major_areas(bucket histogram[], size_t buckets) {
  double_t threshold = avg(histogram, buckets) / 2;

  size_t start = 0;
  uint32_t window_sum = sum(histogram, THRESHOLD_WINDOW_SIZE);
  bool above = ((double_t)window_sum / THRESHOLD_WINDOW_SIZE) > threshold;

  std::vector<activity> res;

  for(size_t i = 1; i < buckets - THRESHOLD_WINDOW_SIZE; i++) {
    window_sum -= histogram[i - 1].count;
    window_sum += histogram[i + THRESHOLD_WINDOW_SIZE - 1].count;
    bool current_above = (double_t)window_sum / THRESHOLD_WINDOW_SIZE > threshold;
    if(current_above != above && i != start) {
      res.push_back({start, i, above});
      above = current_above;
      start = i + 1;
    }
  }

  res.push_back({start, buckets - 1, above});

  for(activity a : res) {
    log("area from %lu to %lu with activity %s\n", a.start, a.end, a.high_activity ? "HIGH" : "LOW");
  }

  return res;
}

uint64_t find_conflict_threshold(void *alloc_start, size_t allocated_pages, size_t step_size) {
  std::vector<uint64_t> samples;
  samples.reserve(THRESHOLD_SAMPLES);

  for(int i = 0; i < THRESHOLD_SAMPLES; i++) {
    auto addr1 = (volatile char*)random_page_addr(alloc_start, allocated_pages, step_size);
    auto addr2 = (volatile char*)random_page_addr(alloc_start, allocated_pages, step_size);
    auto access_time = measure_timing(addr1, addr2);
    samples.push_back(access_time);
  }

  log("created %lu samples", samples.size());

  auto hist = to_hist(samples, N_BUCKETS);
  std::sort(samples.begin(), samples.end());
  log("theoretical threshold for 16 banks is %lu\n", samples[samples.size() * (15.0 / 16.0)]);
  for(size_t i = 0; i < N_BUCKETS; i++) {
    log("hist %lu at %f to %f: %d\n", i, hist[i].ar.lower_bound, hist[i].ar.upper_bound, hist[i].count);
  }

  std::vector<activity> areas = find_major_areas(hist, N_BUCKETS);
  if(areas.size() < 2) {
    log("error: only able to identify %lu areas, expected at least 2.\n", areas.size());
    exit(EXIT_FAILURE);
  }

  size_t i = 0;
  while(i < areas.size() && !areas[i].high_activity) {
    i++;
  }
  while(i < areas.size() && (areas[i].high_activity || areas[i].end - areas[i].start < N_BUCKETS / 10)) {
    i++;
  }

  if(i >= areas.size()) {
    log("unable to identify area of high activity. Something has gone horribly wrong.\n");
    exit(EXIT_FAILURE);
  }

  activity threshold_act = areas[i];
  double_t area_start = hist[threshold_act.start].ar.upper_bound;
  double_t area_end = hist[threshold_act.end].ar.lower_bound;

  uint64_t threshold = area_start + (area_end - area_start) / 2;

  uint32_t above_threshold = 0;
  for(auto sample : samples) {
    if(sample > threshold) {
      above_threshold++;
    }
  }

  log("ratio was %f\n", ((double_t)above_threshold) / samples.size());

  return threshold;
}

enum threshold_failure_t {
  ABOVE, BELOW, INCONSISTENT
};

enum failure_type {
  BANK, INTER_BANK, ROW, CLOSE_ROW, COLUMN, OFFSET
};

typedef struct {
  void *addr1;
  void *addr2;
  uint64_t timing;
  threshold_failure_t type;
  failure_type ftype;
} failure;

typedef struct {
  size_t bank_increment;
  size_t row_increment;
  size_t col_increment;
  size_t offset_increment;
} increment;

typedef struct {
  DRAMAddr *fixed_addr;
  int steps;
  increment inc;
  threshold_failure_t measure_fail_type;
  failure_type scope;
  char *alloc_start;
  size_t alloc_size;
  uint64_t threshold_cycles;
  std::vector<failure> *failures;
} measure_session_conf;

const char* threshold_failure_t_str(threshold_failure_t type) {
  switch(type) {
    case threshold_failure_t::ABOVE:
      return "ABOVE";
    case threshold_failure_t::BELOW:
      return "BELOW";
    case threshold_failure_t::INCONSISTENT:
      return "INCONSISTENT";
  }
  return "UNKNOWN";
}

const char* failure_type_str(failure_type type) {
  switch(type) {
    case failure_type::BANK:
      return "BANK";
    case failure_type::ROW:
      return "ROW";
    case failure_type::COLUMN:
      return "COLUMN";
    case failure_type::CLOSE_ROW:
      return "CLOSE_ROW";
    case failure_type::INTER_BANK:
      return "INTER_BANK";
    case failure_type::OFFSET:
      return "OFFSET";
  }
  return "UNKNOWN";
}

void export_failures(const char* fname, std::vector<failure> *failures) {
  FILE *err_file = fopen(fname, "w+");
  if(err_file == NULL) {
    log("unable to open failed.csv.\n");
    return;
  }

  for(auto f : *failures) {
    fprintf(err_file, "%p;%p;%lu;%s;%s\n", f.addr1, f.addr2, f.timing, threshold_failure_t_str(f.type), failure_type_str(f.ftype));
  }
}

uint64_t run_test(measure_session_conf config) {
  uint64_t n_failed = 0;
  increment inc = config.inc;
  DRAMAddr *fixed = config.fixed_addr;
  size_t offset = 0;
  for(int i = 0; i < config.steps; i++) {
    offset += inc.offset_increment;
    DRAMAddr conflict_addr = fixed->add(inc.bank_increment, inc.row_increment, inc.col_increment);
    char* conflict_virt = (char *)conflict_addr.to_virt() + offset;
    void* fixed_virt = fixed->to_virt();
    if(conflict_virt == fixed_virt) {
      continue;
    }
    if(conflict_virt < config.alloc_start || conflict_virt > config.alloc_start + config.alloc_size) {
      log_err("stopping test type %s for %s because we tried accessing an address outside of the allocated space.\n",
              failure_type_str(config.scope),
              conflict_addr.to_string().c_str());
      break;
    }

    uint64_t time = measure_timing((volatile char *)conflict_virt, (volatile char *)fixed_virt);
    
    bool failed = false;
    if((config.measure_fail_type == threshold_failure_t::ABOVE && time > config.threshold_cycles)
      || (config.measure_fail_type == threshold_failure_t::BELOW && time <= config.threshold_cycles)) {
      
      log_err("[ERR][%s] mapping seems to be wrong between %s and %s as timing (%lu) was %lu %s the calculated threshold (%lu).\n",
              failure_type_str(config.scope),
              fixed->to_string().c_str(),
              conflict_addr.to_string().c_str(),
              time,
              config.measure_fail_type == threshold_failure_t::BELOW ? config.threshold_cycles - time : time - config.threshold_cycles,
              config.measure_fail_type == threshold_failure_t::BELOW ? "below" : "above",
              config.threshold_cycles);
      failed = true;
    } else {
      log("[OK][%s] mapping seems to be correct between %s and %s as timing (%lu) was %lu %s the calculated threshold (%lu).\n",
              failure_type_str(config.scope),
              fixed->to_string().c_str(),
              conflict_addr.to_string().c_str(),
              time,
              config.measure_fail_type == threshold_failure_t::BELOW ? time - config.threshold_cycles : config.threshold_cycles - time,
              config.measure_fail_type == threshold_failure_t::BELOW ? "above" : "below",
              config.threshold_cycles);
    }

    if(failed) {
      config.failures->push_back({ fixed->to_virt(), conflict_virt, time, config.measure_fail_type, config.scope });
      n_failed++;
    }
  }

  return n_failed;
}

uint32_t run_inconsistency_test(DRAMAddr base, int n, int step_size, char* alloc_start, size_t alloc_size, uint64_t threshold_cycles) {
  char *base_virt = (char *)base.to_virt();
  uint32_t n_failed = 0;
  std::vector<failure> failures;
  for(int i = 0; i < n; i += step_size) {
    char *target_virt = base_virt += i;
    if(target_virt > alloc_start + alloc_size) {
      log("stopping inconsistency test as we are wandering outside of the allocated area.");
      break;
    }
    
    DRAMAddr target(target_virt);

    uint64_t time = measure_timing((volatile char *)base_virt, (volatile char *)target_virt);

    threshold_failure_t current = time > threshold_cycles ? threshold_failure_t::ABOVE : threshold_failure_t::BELOW;
    bool failed = false;
    if(current == threshold_failure_t::ABOVE && (target.actual_row() == base.actual_row() || target.actual_bank() != base.actual_bank())) {
      log_err("[ERR][OFFSET] mapping seems to be wrong between %s and %s as the measurement is not consistent with expected timing (should be BELOW). (%s with timing %lu)",
              base.to_string().c_str(),
              target.to_string().c_str(),
              threshold_failure_t_str(current),
              time);
      failed = true;
    } else if(current == threshold_failure_t::BELOW && (target.actual_bank() == base.actual_bank() && target.actual_row() != base.actual_row())) {
      log_err("[ERR][OFFSET] mapping seems to be wrong between %s and %s as the measurement is not consistent with expected timing (should be ABOVE). (%s with timing %lu)",
              base.to_string().c_str(),
              target.to_string().c_str(),
              threshold_failure_t_str(current),
              time);
      failed = true;
    } else {
      log("[OK][OFFSET] mapping seems to be correct between %s and %s as the measurement is consistent. (%s with timing %lu)",
              base.to_string().c_str(),
              target.to_string().c_str(),
              threshold_failure_t_str(current),
              time);
    }
    
    if(failed) {
      failures.push_back({ base_virt, target_virt, time, current, failure_type::OFFSET });
      n_failed++;
    }
  }

  export_failures("inconsistent.csv", &failures);

  return n_failed;
}

void test_col_threshold(DRAMAddr row_base, int n, char *alloc_start, int alloc_size, uint64_t threshold_cycles, std::vector<failure> *failures) {
  size_t max_cols = DRAMConfig::get().columns();
  size_t col_increment = max_cols > n ? max_cols / n : 1;
  increment inc = { 0, 0, col_increment, 0 };
  measure_session_conf config;
  config.scope = failure_type::COLUMN;
  config.measure_fail_type = threshold_failure_t::ABOVE; //accesses on the same column should be fast.
  config.failures = failures;
  config.threshold_cycles = threshold_cycles;
  config.alloc_size = alloc_size;
  config.alloc_start = alloc_start;
  config.inc = inc;
  config.fixed_addr = &row_base;
  config.steps = n;
  
  uint64_t failed = run_test(config);
  if(failed) {
    log_err("column conflict test failed for base %s with %lu detected failures.", row_base.to_string().c_str(), failed);
  }
}

void test_row_threshold(DRAMAddr bank_base, int n, char *alloc_start, int alloc_size, uint64_t threshold_cycles, std::vector<failure> *failures) {
  uint64_t max_rows = DRAMConfig::get().rows();
  uint64_t banks = DRAMConfig::get().banks();
  DRAMAddr fixed = bank_base;
  measure_session_conf config;
  config.steps = n;
  config.alloc_start = alloc_start;
  config.alloc_size = alloc_size;
  config.failures = failures;
  config.threshold_cycles = threshold_cycles;

  for(int i = 0; i < n; i++) {
    DRAMAddr row_conflict_test_addr = fixed;
    measure_session_conf c = config;
    c.inc = { 0, (max_rows - fixed.actual_row()) / n, 0, 0 };
    c.fixed_addr = &fixed;
    c.steps = n - i + 1;
    c.scope = failure_type::ROW;
    c.measure_fail_type = threshold_failure_t::BELOW; //if access is fast, it means we are still on the same row or a different bank.
    uint64_t failed = run_test(c);
    if(failed) {
      log_err("detected %lu failues while measuring row conflicts for %s.", failed, row_conflict_test_addr.to_string().c_str());
    }

    c.inc = { 0, 1, 0, 0 };
    c.steps = 1;
    c.scope = failure_type::CLOSE_ROW;
    c.measure_fail_type = threshold_failure_t::BELOW; //if access is fast, it means we are not hitting a new row or we are hitting a different bank.
    c.fixed_addr = &row_conflict_test_addr;
    failed = run_test(c);
    if(failed) {
      log_err("detected %lu failures while measuring close-row conflict test for %s.", failed, row_conflict_test_addr.to_string().c_str());
    }

    c.inc = { 1, 0, 0, 0 };
    c.steps = banks - 1;
    c.fixed_addr = &row_conflict_test_addr;
    c.measure_fail_type = threshold_failure_t::ABOVE; //if access is slow, we are hitting another row on the SAME bank.
    c.scope = failure_type::INTER_BANK;
    failed = run_test(c);
    if(failed) {
      log_err("detected %lu failures while measuring inter-bank conflict test for %s.", failed, row_conflict_test_addr.to_string().c_str());
    }
    
    test_col_threshold(row_conflict_test_addr, n, alloc_start, alloc_size, threshold_cycles, failures);
    fixed.add_inplace(0, (max_rows / n), 0);
    if(fixed.to_virt() > alloc_start + alloc_size) {
      break;
    }
  }
}

void test_bank_threshold(int n, char *alloc_start, int alloc_size, uint64_t threshold_cycles) {
  uint64_t max_banks = DRAMConfig::get().banks();
  DRAMAddr fixed(alloc_start);
  size_t increment = n > max_banks ? 1 : max_banks / n;
  log("bank increment set to %d\n", increment);
  int max = n > max_banks ? max_banks : n;
  std::vector<failure> failed_addr;
  for(int i = 0; i < max; i += increment) {
    measure_session_conf conf;
    conf.scope = failure_type::BANK;
    conf.measure_fail_type = threshold_failure_t::ABOVE; //if access is slow, we are hitting the same bank.
    conf.fixed_addr = &fixed;
    conf.steps = max;
    conf.inc = { increment, 0, 0, 0 };
    conf.threshold_cycles = threshold_cycles;
    conf.alloc_size = alloc_size;
    conf.alloc_start = alloc_start;
    uint64_t failed = run_test(conf);
    if(failed) {
      log_err("bank conflict test failed %lu times for %s.", failed, fixed.to_string().c_str());
    }
    
    test_row_threshold(fixed, n, alloc_start, alloc_size, threshold_cycles, &failed_addr);
    fixed.add_inplace(increment, 0, 0);
    log("increased bank by %d.\n", increment);
    if(fixed.to_virt() > alloc_start + alloc_size) {
      break;
    }
  }

  export_failures("failures.csv", &failed_addr);
}

bool parse_args(int argc, char *argv[]) {
  if(argc == 1) {
    return true;
  }
  int i = 1;
  do {
    if(strncmp(argv[i], "-v", 2) == 0 || strncmp(argv[i], "--verbose", 2) == 0) {
      verbose = true;
    } else {
      log("unknown option provided (%s). The only valid option is -v or --verbose for now.\n", argv[i]);
      return false;
    }
    i++;
  } while(i < argc);
  return true;
}

int main (int argc, char *argv[]) {
  if(!parse_args(argc, argv)) {
    exit(EXIT_FAILURE);
  }

  log("allocating pages...\n");
  void *alloc_start = allocate(N_PAGES);
  DRAMConfig::select_config(Microarchitecture::AMD_ZEN_3, 1, 4, 4, false);
  DRAMAddr::initialize_mapping(0, (volatile char *)alloc_start);
  
  uint64_t threshold = find_conflict_threshold(alloc_start, N_PAGES, 8192);
  log("determined threshold to be %lu cycles.\n", threshold);

  log("running inconsistency test...");
  //check all bits up to bit 30 for inconsistent behaviour.
  uint64_t inconsistent_addresses = run_inconsistency_test(DRAMAddr(alloc_start), (1 << 30) - 1, 1, (char *)alloc_start, N_PAGES * PAGE_SIZE, threshold);
  log("finished inconsistency test.");
  if(inconsistent_addresses) {
    log_err("found %lu inconsistent address pairs.", inconsistent_addresses);
  }

  test_bank_threshold(750, (char *)alloc_start, N_PAGES * PAGE_SIZE, threshold);

  log("done.\n");
  return 0;
}
