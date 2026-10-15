// Times match_prefix on a 2048-token prompt (128 blocks), all cached.
// Target from PLAN.md: under 10 microseconds per call.
// The ns/block number is the full per-block cost: hash + probe + memcmp +
// incref + decref.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "block_manager.hpp"
#include "radix_tree.hpp"

using namespace prefixserve;
using Clock = std::chrono::steady_clock;

int main() {
  constexpr int32_t kTokens = 2048;
  constexpr int kIters = 200000;

  BlockManager bm(1 << 16);
  RadixTree tree(bm);

  std::vector<int32_t> toks(kTokens);
  for (int32_t i = 0; i < kTokens; ++i) toks[static_cast<size_t>(i)] = (i * 2654435761u) % 50000;

  std::vector<BlockId> blocks;
  for (int32_t i = 0; i < kTokens / kBlockSize; ++i) blocks.push_back(*bm.allocate());
  if (tree.insert(toks.data(), kTokens, blocks.data(), static_cast<int64_t>(blocks.size()), nullptr) !=
      Status::kOk) {
    std::fprintf(stderr, "insert failed\n");
    return 1;
  }

  // Lots of unrelated cached prompts so the hash table isn't tiny and hot.
  for (int p = 0; p < 300; ++p) {
    std::vector<int32_t> t(256);
    for (size_t i = 0; i < t.size(); ++i) t[i] = static_cast<int32_t>(100000 + p * 1000 + i);
    std::vector<BlockId> b;
    for (int i = 0; i < 16; ++i) b.push_back(*bm.allocate());
    (void)tree.insert(t.data(), 256, b.data(), 16, nullptr);
    for (BlockId x : b) (void)bm.decref(x);
  }

  std::vector<BlockId> out;
  out.reserve(static_cast<size_t>(bm.num_blocks()));

  for (int i = 0; i < 1000; ++i) {  // warm up
    tree.match_prefix(toks.data(), kTokens, out);
    for (BlockId b : out) (void)bm.decref(b);
  }

  const auto t0 = Clock::now();
  int64_t sink = 0;
  for (int i = 0; i < kIters; ++i) {
    sink += tree.match_prefix(toks.data(), kTokens, out);
    for (BlockId b : out) (void)bm.decref(b);
  }
  const auto t1 = Clock::now();

  const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / kIters;
  std::printf("match_prefix(2048 tokens, 128 blocks) + decref: %.0f ns/call (%.1f ns/block)\n", ns,
              ns / 128.0);
  std::printf("target: < 10000 ns  -> %s\n", ns < 10000 ? "PASS" : "FAIL");
  return (sink > 0 && ns < 10000) ? 0 : 1;
}
