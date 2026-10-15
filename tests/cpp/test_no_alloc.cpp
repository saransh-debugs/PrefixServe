// Proves the hot path never touches the heap.
//
// We replace global operator new with a counting version and check the count
// stays at 0 across match_prefix / insert / evict / allocate / incref /
// decref.
//
// Why not ASan or Valgrind: they find memory *bugs* (use-after-free,
// overflow). They don't report a perfectly valid allocation that happens to
// be slow. Only counting catches "a vector grew in the hot path".

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <new>
#include <vector>

#include "block_manager.hpp"
#include "radix_tree.hpp"

namespace {

std::atomic<bool> g_counting{false};
std::atomic<long> g_allocs{0};

void* counted_alloc(std::size_t n) {
  if (g_counting.load(std::memory_order_relaxed)) g_allocs.fetch_add(1, std::memory_order_relaxed);
  if (void* p = std::malloc(n == 0 ? 1 : n)) return p;
  throw std::bad_alloc();
}

void* counted_aligned_alloc(std::size_t n, std::align_val_t al) {
  if (g_counting.load(std::memory_order_relaxed)) g_allocs.fetch_add(1, std::memory_order_relaxed);
  void* p = nullptr;
  const auto a = std::max(static_cast<std::size_t>(al), sizeof(void*));
  if (posix_memalign(&p, a, n == 0 ? 1 : n) != 0) throw std::bad_alloc();
  return p;
}

// Counts allocations while in scope. Read count() before any EXPECT_*
// because a failing gtest assertion allocates its message.
class AllocCounter {
 public:
  AllocCounter() {
    g_allocs.store(0);
    g_counting.store(true);
  }
  ~AllocCounter() { g_counting.store(false); }
  long stop() {
    g_counting.store(false);
    return g_allocs.load();
  }
};

}  // namespace

void* operator new(std::size_t n) { return counted_alloc(n); }
void* operator new[](std::size_t n) { return counted_alloc(n); }
void* operator new(std::size_t n, std::align_val_t a) { return counted_aligned_alloc(n, a); }
void* operator new[](std::size_t n, std::align_val_t a) { return counted_aligned_alloc(n, a); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  try { return counted_alloc(n); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
  try { return counted_alloc(n); } catch (...) { return nullptr; }
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

using namespace prefixserve;

// Without this, a broken counter would make every other test pass.
TEST(NoAlloc, CounterActuallyCounts) {
  AllocCounter c;
  auto* v = new std::vector<int>(100);
  const long n = c.stop();
  delete v;
  EXPECT_GE(n, 2);
}

TEST(NoAlloc, BlockManagerHotPath) {
  BlockManager bm(256);
  AllocCounter c;
  for (int round = 0; round < 100; ++round) {
    BlockId ids[256];
    int got = 0;
    while (auto b = bm.allocate()) ids[got++] = *b;
    for (int i = 0; i < got; ++i) (void)bm.incref(ids[i]);
    for (int i = 0; i < got; ++i) (void)bm.decref(ids[i]);
    for (int i = 0; i < got; ++i) (void)bm.decref(ids[i]);
  }
  const long n = c.stop();
  EXPECT_EQ(n, 0);
}

TEST(NoAlloc, RadixTreeHotPath) {
  constexpr int32_t kBlocks = 512;
  BlockManager bm(kBlocks);
  RadixTree tree(bm);

  // Setup is allowed to allocate.
  std::vector<BlockId> out;
  out.reserve(kBlocks);
  std::vector<BlockId> fresh;
  fresh.reserve(kBlocks);
  std::vector<std::vector<int32_t>> prompts;
  for (int p = 0; p < 8; ++p) {
    std::vector<int32_t> t(static_cast<size_t>(16 * (4 + p)));
    for (size_t i = 0; i < t.size(); ++i) t[i] = static_cast<int32_t>((p % 3) * 7 + i);
    prompts.push_back(std::move(t));
  }

  AllocCounter c;
  for (int round = 0; round < 200; ++round) {
    for (const auto& toks : prompts) {
      const auto n = static_cast<int64_t>(toks.size());
      tree.match_prefix(toks.data(), n, out);
      fresh.clear();
      for (BlockId b : out) fresh.push_back(b);
      const auto needed = static_cast<size_t>(n / kBlockSize);
      while (fresh.size() < needed) {
        auto b = bm.allocate();
        if (!b) {
          if (tree.evict(8) == 0) break;
          continue;
        }
        fresh.push_back(*b);
      }
      if (fresh.size() == needed) {
        (void)tree.insert(toks.data(), n, fresh.data(), static_cast<int64_t>(fresh.size()), nullptr);
      }
      for (BlockId b : fresh) (void)bm.decref(b);
    }
    tree.evict(16);
  }
  const long n = c.stop();
  EXPECT_EQ(n, 0) << "hot path allocated " << n << " times";
}
