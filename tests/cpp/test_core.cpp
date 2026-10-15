// Core correctness tests. Built with ASan + UBSan (see CMakeLists.txt).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "block_manager.hpp"
#include "radix_tree.hpp"

using namespace prefixserve;

namespace {

void expect_ok(const BlockManager& bm, const RadixTree& tree) {
  std::string why;
  ASSERT_TRUE(bm.check_invariants(&why)) << why;
  ASSERT_TRUE(tree.check_invariants(&why)) << why;
}

std::vector<int32_t> seq(int32_t n, int32_t start = 0) {
  std::vector<int32_t> v(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = start + i;
  return v;
}

// Plays one request's prefill: reuse cached blocks, allocate the rest,
// cache the result. Returns every block the request holds a ref on.
std::vector<BlockId> prefill(BlockManager& bm, RadixTree& tree, const std::vector<int32_t>& toks) {
  std::vector<BlockId> blocks;
  blocks.reserve(static_cast<size_t>(bm.num_blocks()));
  tree.match_prefix(toks.data(), static_cast<int64_t>(toks.size()), blocks);
  const size_t needed = (toks.size() + kBlockSize - 1) / kBlockSize;
  while (blocks.size() < needed) {
    auto b = bm.allocate();
    EXPECT_TRUE(b.has_value());
    if (!b) break;
    blocks.push_back(*b);
  }
  EXPECT_EQ(tree.insert(toks.data(), static_cast<int64_t>(toks.size()), blocks.data(),
                        static_cast<int64_t>(blocks.size()), nullptr),
            Status::kOk);
  return blocks;
}

void release(BlockManager& bm, const std::vector<BlockId>& blocks) {
  for (BlockId b : blocks) EXPECT_EQ(bm.decref(b), Status::kOk);
}

int64_t match_len(RadixTree& tree, BlockManager& bm, const std::vector<int32_t>& toks) {
  std::vector<BlockId> out;
  const int64_t n = tree.match_prefix(toks.data(), static_cast<int64_t>(toks.size()), out);
  release(bm, out);
  return n;
}

}  // namespace

// ---------------- BlockManager ----------------

TEST(BlockManager, RejectsBadSize) {
  EXPECT_THROW(BlockManager(0), std::invalid_argument);
  EXPECT_THROW(BlockManager(-5), std::invalid_argument);
  EXPECT_THROW(BlockManager(kMaxBlocks + 1), std::invalid_argument);
}

TEST(BlockManager, AllocatesUntilEmpty) {
  BlockManager bm(4);
  for (BlockId want = 0; want < 4; ++want) {
    auto b = bm.allocate();
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(*b, want);
    EXPECT_EQ(bm.ref_count(*b), 1);
  }
  EXPECT_FALSE(bm.allocate().has_value());
  EXPECT_EQ(bm.num_free(), 0);
}

TEST(BlockManager, RefCountingFreesAtZero) {
  BlockManager bm(2);
  const BlockId b = *bm.allocate();
  EXPECT_EQ(bm.incref(b), Status::kOk);
  EXPECT_EQ(bm.ref_count(b), 2);
  EXPECT_EQ(bm.decref(b), Status::kOk);
  EXPECT_EQ(bm.num_free(), 1);
  EXPECT_EQ(bm.decref(b), Status::kOk);
  EXPECT_EQ(bm.num_free(), 2);
  std::string why;
  EXPECT_TRUE(bm.check_invariants(&why)) << why;
}

TEST(BlockManager, DoubleFreeIsRejectedAndHarmless) {
  BlockManager bm(2);
  const BlockId b = *bm.allocate();
  ASSERT_EQ(bm.decref(b), Status::kOk);
  EXPECT_EQ(bm.decref(b), Status::kBlockNotAllocated);
  EXPECT_EQ(bm.num_free(), 2);  // not pushed twice
  EXPECT_EQ(bm.incref(b), Status::kBlockNotAllocated);
  std::string why;
  EXPECT_TRUE(bm.check_invariants(&why)) << why;
}

TEST(BlockManager, InvalidIdsRejected) {
  BlockManager bm(2);
  EXPECT_EQ(bm.incref(-1), Status::kInvalidBlockId);
  EXPECT_EQ(bm.decref(2), Status::kInvalidBlockId);
  EXPECT_EQ(bm.ref_count(99), -1);
}

TEST(BlockManager, LifoReuse) {
  BlockManager bm(8);
  const BlockId a = *bm.allocate();
  (void)*bm.allocate();
  ASSERT_EQ(bm.decref(a), Status::kOk);
  EXPECT_EQ(*bm.allocate(), a);
}

// ---------------- RadixTree ----------------

TEST(RadixTree, EmptyTreeMatchesNothing) {
  BlockManager bm(8);
  RadixTree tree(bm);
  EXPECT_EQ(match_len(tree, bm, seq(64)), 0);
  expect_ok(bm, tree);
}

TEST(RadixTree, InsertThenMatchFullBlocksOnly) {
  BlockManager bm(8);
  RadixTree tree(bm);
  const auto toks = seq(40);  // 2 full blocks + 8 leftover tokens
  auto held = prefill(bm, tree, toks);
  ASSERT_EQ(held.size(), 3u);
  EXPECT_EQ(tree.num_cached_blocks(), 2);  // partial block not cached
  EXPECT_FALSE(tree.is_cached(held[2]));

  std::vector<BlockId> out;
  EXPECT_EQ(tree.match_prefix(toks.data(), 40, out), 32);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0], held[0]);
  EXPECT_EQ(out[1], held[1]);
  // tree + first request + this match
  EXPECT_EQ(bm.ref_count(held[0]), 3);
  release(bm, out);
  release(bm, held);
  EXPECT_EQ(bm.ref_count(held[0]), 1);  // only the tree
  expect_ok(bm, tree);
}

TEST(RadixTree, PartialBlockIsNeverCached) {
  BlockManager bm(4);
  RadixTree tree(bm);
  auto held = prefill(bm, tree, seq(15));
  EXPECT_EQ(tree.num_cached_blocks(), 0);
  release(bm, held);
  EXPECT_EQ(bm.num_free(), 4);
  expect_ok(bm, tree);
}

// The case a first-token child map gets wrong: two sibling blocks that
// start with the same token and differ later in the block.
TEST(RadixTree, SiblingsSharingFirstTokenAreDistinct) {
  BlockManager bm(16);
  RadixTree tree(bm);
  auto a = seq(32);        // block0 = 0..15, block1 = 16..31
  auto b = a;
  b[17] = 999;             // block1 still starts with 16, differs at token 17

  auto held_a = prefill(bm, tree, a);
  EXPECT_EQ(match_len(tree, bm, b), 16) << "must not reuse A's second block for B";

  auto held_b = prefill(bm, tree, b);
  EXPECT_EQ(tree.num_cached_blocks(), 3);  // shared block0, two different block1s
  EXPECT_NE(held_a[1], held_b[1]);
  EXPECT_EQ(held_a[0], held_b[0]);

  std::vector<BlockId> out;
  ASSERT_EQ(tree.match_prefix(b.data(), 32, out), 32);
  EXPECT_EQ(out[1], held_b[1]);
  release(bm, out);
  release(bm, held_a);
  release(bm, held_b);
  expect_ok(bm, tree);
}

TEST(RadixTree, SameTokensAtDifferentDepthAreDifferentNodes) {
  BlockManager bm(16);
  RadixTree tree(bm);
  auto chunk = seq(16, 100);
  std::vector<int32_t> twice = chunk;
  twice.insert(twice.end(), chunk.begin(), chunk.end());
  auto held = prefill(bm, tree, twice);
  EXPECT_EQ(tree.num_cached_blocks(), 2);
  EXPECT_EQ(match_len(tree, bm, chunk), 16);
  EXPECT_EQ(match_len(tree, bm, twice), 32);
  release(bm, held);
  expect_ok(bm, tree);
}

TEST(RadixTree, DuplicateContentKeepsTreeCopy) {
  BlockManager bm(8);
  RadixTree tree(bm);
  const auto toks = seq(32);
  auto first = prefill(bm, tree, toks);

  // A second request computed the same prompt into its own blocks
  // (e.g. both arrived before either finished).
  std::vector<BlockId> mine = {*bm.allocate(), *bm.allocate()};
  int32_t added = -1;
  EXPECT_EQ(tree.insert(toks.data(), 32, mine.data(), 2, &added), Status::kOk);
  EXPECT_EQ(added, 0);
  EXPECT_FALSE(tree.is_cached(mine[0]));
  EXPECT_EQ(bm.ref_count(mine[0]), 1);  // still only the caller's ref

  release(bm, mine);
  release(bm, first);
  expect_ok(bm, tree);
}

TEST(RadixTree, InsertRejectsBadInputWithoutChanges) {
  BlockManager bm(8);
  RadixTree tree(bm);
  auto held = prefill(bm, tree, seq(16));
  const auto other = seq(32, 500);

  // Same block in two places.
  std::vector<BlockId> reuse = {held[0], *bm.allocate()};
  EXPECT_EQ(tree.insert(other.data(), 32, reuse.data(), 2, nullptr), Status::kBlockAlreadyCached);

  // Same new id twice in one call. Must roll back the mark on the first one.
  const BlockId fresh = *bm.allocate();
  std::vector<BlockId> dup = {fresh, fresh};
  EXPECT_EQ(tree.insert(other.data(), 32, dup.data(), 2, nullptr), Status::kBlockAlreadyCached);
  EXPECT_FALSE(tree.is_cached(fresh));

  // Block the caller doesn't own.
  std::vector<BlockId> freed = {7, 6};
  ASSERT_EQ(bm.ref_count(7), 0);
  EXPECT_EQ(tree.insert(other.data(), 32, freed.data(), 2, nullptr), Status::kBlockNotAllocated);

  // Out of range, and too few blocks.
  std::vector<BlockId> bad = {fresh, 1000};
  EXPECT_EQ(tree.insert(other.data(), 32, bad.data(), 2, nullptr), Status::kInvalidBlockId);
  EXPECT_EQ(tree.insert(other.data(), 32, &fresh, 1, nullptr), Status::kInvalidArgument);

  EXPECT_EQ(tree.num_cached_blocks(), 1);
  expect_ok(bm, tree);

  // After all that, a valid insert with `fresh` still works.
  std::vector<BlockId> good = {fresh, reuse[1]};
  int32_t added = 0;
  EXPECT_EQ(tree.insert(other.data(), 32, good.data(), 2, &added), Status::kOk);
  EXPECT_EQ(added, 2);
  release(bm, good);
  release(bm, held);
  expect_ok(bm, tree);
}

TEST(RadixTree, EvictsLeastRecentlyUsedFirst) {
  BlockManager bm(8);
  RadixTree tree(bm);
  const auto a = seq(16, 0), b = seq(16, 100), c = seq(16, 200);
  release(bm, prefill(bm, tree, a));
  release(bm, prefill(bm, tree, b));
  release(bm, prefill(bm, tree, c));
  EXPECT_EQ(match_len(tree, bm, a), 16);  // a is now the newest

  EXPECT_EQ(tree.evict(1), 1);
  EXPECT_EQ(match_len(tree, bm, b), 0) << "b was oldest";
  EXPECT_EQ(match_len(tree, bm, a), 16);
  EXPECT_EQ(match_len(tree, bm, c), 16);
  expect_ok(bm, tree);
}

TEST(RadixTree, EvictSkipsBlocksInUse) {
  BlockManager bm(8);
  RadixTree tree(bm);
  const auto a = seq(16, 0), b = seq(16, 100);
  auto held_a = prefill(bm, tree, a);     // still running
  release(bm, prefill(bm, tree, b));

  EXPECT_EQ(tree.evict(8), 1);            // only b could go
  EXPECT_EQ(match_len(tree, bm, a), 16);
  expect_ok(bm, tree);

  release(bm, held_a);
  EXPECT_EQ(tree.evict(8), 1);
  EXPECT_EQ(bm.num_free(), 8);
  expect_ok(bm, tree);
}

TEST(RadixTree, EvictWalksUpTheChain) {
  BlockManager bm(8);
  RadixTree tree(bm);
  release(bm, prefill(bm, tree, seq(48)));  // 3-block chain
  EXPECT_EQ(tree.evict(10), 3);             // parents become leaves as we go
  EXPECT_EQ(tree.num_cached_blocks(), 0);
  EXPECT_EQ(bm.num_free(), 8);
  expect_ok(bm, tree);
}

TEST(RadixTree, DestructorGivesBackRefs) {
  BlockManager bm(8);
  {
    RadixTree tree(bm);
    release(bm, prefill(bm, tree, seq(64)));
    EXPECT_EQ(bm.num_free(), 4);
  }
  EXPECT_EQ(bm.num_free(), 8);
  std::string why;
  EXPECT_TRUE(bm.check_invariants(&why)) << why;
}

// Random requests, cancellations and evictions. After every step:
//   - both structures pass their own invariant checks
//   - every block's ref_count == (1 if cached) + (refs held by live requests)
// At the end, everything goes back to the pool.
TEST(RadixTree, RandomizedRefCountModel) {
  constexpr int32_t kBlocks = 96;
  BlockManager bm(kBlocks);
  RadixTree tree(bm);
  std::mt19937 rng(12345);

  // Few distinct prefixes + random tails => lots of sharing and branching.
  std::vector<std::vector<int32_t>> prefixes;
  for (int p = 0; p < 4; ++p) prefixes.push_back(seq(16 * (1 + p), 1000 * p));

  std::vector<std::vector<BlockId>> live;

  auto check_model = [&] {
    std::map<BlockId, int32_t> expected;
    for (const auto& r : live)
      for (BlockId b : r) ++expected[b];
    for (BlockId b = 0; b < kBlocks; ++b) {
      const int32_t want = expected[b] + (tree.is_cached(b) ? 1 : 0);
      ASSERT_EQ(bm.ref_count(b), want) << "block " << b;
    }
  };

  for (int step = 0; step < 4000; ++step) {
    const int op = static_cast<int>(rng() % 10);
    if (op < 5) {
      // New request: shared prefix + random tail.
      auto toks = prefixes[rng() % prefixes.size()];
      const int tail = static_cast<int>(rng() % 40);
      for (int i = 0; i < tail; ++i) toks.push_back(static_cast<int32_t>(rng() % 3));
      std::vector<BlockId> blocks;
      blocks.reserve(kBlocks);
      tree.match_prefix(toks.data(), static_cast<int64_t>(toks.size()), blocks);
      const size_t needed = (toks.size() + kBlockSize - 1) / kBlockSize;
      bool ok = true;
      while (blocks.size() < needed) {
        auto b = bm.allocate();
        if (!b) {
          if (tree.evict(static_cast<int32_t>(needed - blocks.size())) == 0) { ok = false; break; }
          continue;
        }
        blocks.push_back(*b);
      }
      if (!ok) {
        release(bm, blocks);  // can't fit: give back what we took
      } else {
        ASSERT_EQ(tree.insert(toks.data(), static_cast<int64_t>(toks.size()), blocks.data(),
                              static_cast<int64_t>(blocks.size()), nullptr),
                  Status::kOk);
        live.push_back(std::move(blocks));
      }
    } else if (op < 9 && !live.empty()) {
      const size_t i = rng() % live.size();
      release(bm, live[i]);
      live.erase(live.begin() + static_cast<std::ptrdiff_t>(i));
    } else {
      tree.evict(static_cast<int32_t>(rng() % 8));
    }
    expect_ok(bm, tree);
    check_model();
    if (HasFatalFailure()) return;
  }

  for (const auto& r : live) release(bm, r);
  live.clear();
  tree.evict(kBlocks);
  EXPECT_EQ(tree.num_cached_blocks(), 0);
  EXPECT_EQ(bm.num_free(), kBlocks) << "leaked blocks";
  expect_ok(bm, tree);
}
