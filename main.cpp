#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <emmintrin.h>
#include <immintrin.h>
#include <random>
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

void* allocate(size_t n_pages) {
  uint64_t huge_map = 0;
  if(PAGE_SIZE == PAGE_1GB) {
    huge_map = MAP_HUGE_1GB;
  } else if(PAGE_SIZE == PAGE_2MB) {
    huge_map = MAP_HUGE_2MB;
  }
  void* res = mmap(NULL, n_pages * PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE | MAP_HUGETLB | huge_map | MAP_POPULATE, -1, 0);
  if(res == MAP_FAILED) {
    printf("memory allocation failed (%s)\n", strerror(errno));
    exit(EXIT_FAILURE);
  }
  memset(res, 0x42, n_pages * PAGE_SIZE);
  printf("allocated %lu bytes of memory.\n", n_pages * PAGE_SIZE);
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

  printf("measuring timing between %s and %s\n", DRAMAddr((void *)addr1).to_string().c_str(), DRAMAddr((void *)addr2).to_string().c_str());

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
    printf("area from %lu to %lu with activity %s\n", a.start, a.end, a.high_activity ? "HIGH" : "LOW");
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

  printf("created %lu samples", samples.size());

  auto hist = to_hist(samples, N_BUCKETS);
  std::sort(samples.begin(), samples.end());
  printf("theoretical threshold for 16 banks is %lu\n", samples[samples.size() * (15.0 / 16.0)]);
  for(size_t i = 0; i < N_BUCKETS; i++) {
    printf("hist %lu at %f to %f: %d\n", i, hist[i].ar.lower_bound, hist[i].ar.upper_bound, hist[i].count);
  }

  std::vector<activity> areas = find_major_areas(hist, N_BUCKETS);
  if(areas.size() < 2) {
    printf("error: only able to identify %lu areas, expected at least 2.\n", areas.size());
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
    printf("unable to identify area of high activity. Something has gone horribly wrong.\n");
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

  printf("ratio was %f\n", ((double_t)above_threshold) / samples.size());

  return threshold;
}

void test_col_threshold(DRAMAddr row_base, int n, char *alloc_start, int alloc_size, uint64_t threshold_cycles) {
  DRAMAddr col_fixed = row_base;
  DRAMAddr col_conflict_test_addr = row_base;
  for(int j = 0; j < n; j++) {
    col_conflict_test_addr.add_inplace(0, 0, DRAMConfig::get().columns() / n);
    printf("testing for same-row access between %s and %s. This should result in no conflict being detected.\n", 
           col_fixed.to_string().c_str(), 
           col_conflict_test_addr.to_string().c_str());

    if(col_conflict_test_addr.to_virt() > alloc_start + alloc_size) {
      printf("stopping test as address %s is outside of the allocated area.\n", col_conflict_test_addr.to_string().c_str());
      break;
    }

    uint64_t time = measure_timing((volatile char *)col_fixed.to_virt(), (volatile char *)col_conflict_test_addr.to_virt());
    if(time <= threshold_cycles) {
      printf("[OK] address %s seems to be in the same row as it does not the row threshold (timing: %lu)\n", 
             col_conflict_test_addr.to_string().c_str(), 
             time);
    } else {
      printf("[ERR] address mapping seems to be wrong for address %s as it is %lu below the threshold (defined as %lu). Timing was %lu.\n", 
             col_conflict_test_addr.to_string().c_str(), 
             threshold_cycles - time, 
             threshold_cycles, 
             time);
    }
  }
}

void test_row_threshold(DRAMAddr bank_base, int n, char *alloc_start, int alloc_size, uint64_t threshold_cycles) {
  uint64_t max_rows = DRAMConfig::get().rows();
  DRAMAddr fixed = bank_base;
  for(int i = 0; i < n; i++) {
    DRAMAddr row_conflict_test_addr = fixed;
    for(int j = i + 1; j < n; j++) {
      row_conflict_test_addr.add_inplace(0, (max_rows - fixed.actual_row()) / n, 0);
      printf("testing address %s against %s. This should result in a row conflict.\n", row_conflict_test_addr.to_string().c_str(), fixed.to_string().c_str());
      if(row_conflict_test_addr.to_virt() > alloc_start + alloc_size) {
        printf("stopping because test address exceeds the allocated space (%p)\n", alloc_start + alloc_size);
        break;
      }
      uint64_t time = measure_timing((volatile char *)fixed.to_virt(), (volatile char *)row_conflict_test_addr.to_virt());
      if(time > threshold_cycles) {
        printf("[OK] address %s seems to be in another row as it exceeds the row threshold (%lu) by %lu (timing: %lu)\n", 
               row_conflict_test_addr.to_string().c_str(), 
               threshold_cycles, 
               time - threshold_cycles, 
               time);
      } else {
        printf("[ERR] address mapping seems to be wrong for address %s as it is %lu below the threshold (defined as %lu). Timing was %lu.\n", 
               row_conflict_test_addr.to_string().c_str(), 
               threshold_cycles - time, 
               threshold_cycles, 
               time);
      }
      test_col_threshold(row_conflict_test_addr, n, alloc_start, alloc_size, threshold_cycles);
    }
    fixed.add_inplace(0, (max_rows / n), 0);
    if(fixed.to_virt() > alloc_start + alloc_size) {
      break;
    }
  }
}

void test_bank_threshold(int n, char *alloc_start, int alloc_size, uint64_t threshold_cycles) {
  uint64_t max_banks = DRAMConfig::get().banks();
  DRAMAddr fixed(alloc_start);
  int increment = n > max_banks ? 1 : max_banks / n;
  printf("bank increment set to %d\n", increment);

  for(int i = 0; i < n; i += increment) {
    DRAMAddr bank_conflict_test_addr = fixed;
    for(int j = i + 1; j < n; j += increment) {
      bank_conflict_test_addr.add_inplace(increment, 0, 0);
      printf("testing address %s against %s. This should result in fast timing.\n", bank_conflict_test_addr.to_string().c_str(), fixed.to_string().c_str());
      if(bank_conflict_test_addr.to_virt() > alloc_start + alloc_size) {
        printf("stopping because test address exceeds the allocated space (%p)\n", alloc_start + alloc_size);
        break;
      }
      uint64_t time = measure_timing((volatile char *)fixed.to_virt(), (volatile char *)bank_conflict_test_addr.to_virt());
      if(time < threshold_cycles) {
        printf("[OK][BANK] address %s seems to be in a different bank as its access time is below the conflict threshold (%lu) (timing: %lu)\n", 
               bank_conflict_test_addr.to_string().c_str(), 
               threshold_cycles, 
               time);
      } else {
        printf("[ERR][BANK] address %s seems to be in the same bank as it is %lu above the threshold (defined as %lu). Timing was %lu.\n", 
               bank_conflict_test_addr.to_string().c_str(), 
               time - threshold_cycles, 
               threshold_cycles, 
               time);
        break;
      }
    }
    test_row_threshold(fixed, n, alloc_start, alloc_size, threshold_cycles);
    fixed.add_inplace(increment, 0, 0);
    printf("increased bank by %d.\n", increment);
    if(fixed.to_virt() > alloc_start + alloc_size) {
      break;
    }
  }
}

int main (int argc, char *argv[]) {
  printf("allocating pages...\n");
  void *alloc_start = allocate(N_PAGES);
  DRAMConfig::select_config(Microarchitecture::AMD_ZEN_3, 1, 4, 4, false);
  DRAMAddr::initialize_mapping(0, (volatile char *)alloc_start);
  
  uint64_t threshold = find_conflict_threshold(alloc_start, N_PAGES, 8192);
  printf("determined threshold to be %lu cycles.\n", threshold);
  test_bank_threshold(50, (char *)alloc_start, N_PAGES * PAGE_SIZE, threshold);

  printf("done.\n");
  return 0;
}
