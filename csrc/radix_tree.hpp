#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "block_manager.hpp"

namespace prefixserve {

// Prefix cache over KV blocks (RadixAttention-style).
//
// Each node owns exactly one full block (16 tokens). A path from the root
// spells out a cached prompt prefix, one block at a time.
//
// Why one node per block instead of compressed edges of any length:
//   KV memory is shared a whole block at a time. An edge that ends halfway
//   through a block can't share that block anyway, so longer edges only add
//   split/merge bookkeeping. One node per block also means
//   node count <= block count, so every array is sized once, up front.
//
// Why children are found by hashing all 16 tokens, not by the first token:
//   two sibling blocks can start with the same token and differ later.
//   A first-token key would hand back the wrong block, and the model would
//   attend over someone else's KV.
//
// Ref counting: the tree holds 1 ref on each block it caches. Each request
// using a block holds 1 more. So ref_count == 1 means "only the tree holds
// it" and the block is safe to evict. Ref counts live only in the
// BlockManager. Nodes don't keep a second copy that could drift.
//
// Nothing in match_prefix / insert / evict allocates. All storage is sized
// in the constructor. tests/cpp/test_no_alloc.cpp checks this.
//
// Not thread-safe. One asyncio task owns it.
class RadixTree {
 public:
  using NodeId = int32_t;

  // `bm` must outlive the tree. The tree gives its refs back in ~RadixTree.
  // `hash_seed` picks the random hash keys. Default: a fresh random seed.
  // Tests can pass a fixed one.
  explicit RadixTree(BlockManager& bm, uint64_t hash_seed = random_seed());
  ~RadixTree();

  RadixTree(const RadixTree&) = delete;
  RadixTree& operator=(const RadixTree&) = delete;
  RadixTree(RadixTree&&) = delete;
  RadixTree& operator=(RadixTree&&) = delete;

  // Finds the longest cached prefix of tokens[0, n), in whole blocks.
  // Writes the matched block ids to `out` (cleared first) and takes 1 ref on
  // each for the caller. The caller must decref them when done.
  // Returns matched token count (always a multiple of kBlockSize).
  // Allocation-free when out.capacity() >= max_blocks().
  int64_t match_prefix(const int32_t* tokens, int64_t n,
                       std::vector<BlockId>& out) noexcept;

  // Caches the full blocks of tokens[0, n). blocks[i] holds the KV for
  // tokens [16*i, 16*i + 16). A trailing partial block is ignored.
  //
  // Blocks already cached (same tokens, same position) are skipped: the tree
  // keeps its copy, the caller keeps its own. New blocks get 1 tree ref.
  //
  // All-or-nothing: every new block is checked before anything changes.
  // `added` (optional) gets the number of new nodes.
  Status insert(const int32_t* tokens, int64_t n, const BlockId* blocks,
                int64_t num_blocks, int32_t* added) noexcept;

  // Frees up to `want` blocks, least recently used first. Only removes
  // leaves whose block nobody else holds (ref_count == 1). Returns how many
  // were freed, which can be less than `want`.
  int32_t evict(int32_t want) noexcept;

  [[nodiscard]] int32_t num_cached_blocks() const noexcept;
  [[nodiscard]] int32_t max_blocks() const noexcept { return bm_.num_blocks(); }
  [[nodiscard]] bool is_cached(BlockId id) const noexcept;

  // O(n) full consistency check. Tests only.
  bool check_invariants(std::string* why) const;

 private:
  static constexpr NodeId kRoot = 0;
  static constexpr NodeId kNoNode = -1;

  struct Node {
    NodeId parent = kNoNode;
    BlockId block = kNoBlock;  // kNoBlock => node is free (or the root)
    int32_t num_children = 0;
    int32_t heap_pos = -1;     // index in heap_, -1 if not in the heap
    uint64_t last_access = 0;  // logical clock, not wall time
  };

  // A block's 16 tokens = 64 bytes = one cache line. alignas makes sure a
  // compare never straddles two lines. (C++17 vector honors over-alignment.)
  struct alignas(64) TokenChunk {
    int32_t t[kBlockSize];
  };
  static_assert(sizeof(TokenChunk) == 64, "one chunk must be one cache line");

  // One flat hash table for every parent -> child edge in the tree.
  // Why not a std::unordered_map per node: each map allocates on insert and
  // scatters its buckets across the heap. One open-addressing table, sized
  // once at <= 50% load, never allocates and keeps probes in a few lines.
  struct Slot {
    uint64_t hash = 0;
    NodeId node = kNoNode;  // kNoNode => empty
  };

  static uint64_t random_seed();
  uint64_t hash_chunk(NodeId parent, const int32_t* tokens) const noexcept;

  int64_t find_slot(NodeId parent, const int32_t* tokens, uint64_t h) const noexcept;
  void table_insert(uint64_t h, NodeId node) noexcept;
  void table_erase(size_t slot) noexcept;

  void remove_leaf(NodeId leaf) noexcept;

  // Min-heap of evictable leaves, ordered by last_access.
  // Why each node stores its heap position: when a leaf is touched or gains
  // a child, we fix or remove it in O(log n). A plain std::priority_queue
  // can't do that, so you'd push duplicates and the heap would grow without
  // bound between evictions.
  bool lru_before(NodeId a, NodeId b) const noexcept;
  void heap_place(size_t pos, NodeId id) noexcept;
  void heap_sift_up(size_t pos) noexcept;
  void heap_sift_down(size_t pos) noexcept;
  void heap_push(NodeId id) noexcept;
  void heap_remove(NodeId id) noexcept;

  BlockManager& bm_;
  int32_t capacity_;  // nodes including the root = num_blocks + 1
  uint32_t hash_key_[kBlockSize];
  uint64_t clock_ = 0;

  std::vector<Node> nodes_;
  std::vector<TokenChunk> tokens_;      // tokens_[node]
  std::vector<NodeId> free_nodes_;
  std::vector<Slot> table_;
  size_t mask_ = 0;
  std::vector<NodeId> heap_;
  std::vector<NodeId> pinned_;          // evict() scratch, reserved
  std::vector<uint8_t> block_cached_;   // block_cached_[block] = tree owns it
};

}  // namespace prefixserve
