#include "radix_tree.hpp"

#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace prefixserve {

namespace {

// splitmix64 finalizer. Cheap and mixes every input bit into every output bit.
inline uint64_t mix64(uint64_t x) noexcept {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

}  // namespace

uint64_t RadixTree::random_seed() {
  std::random_device rd;
  return (static_cast<uint64_t>(rd()) << 32) ^ rd();
}

RadixTree::RadixTree(BlockManager& bm, uint64_t hash_seed)
    : bm_(bm), capacity_(bm.num_blocks() + 1) {
  const auto cap = static_cast<size_t>(capacity_);

  uint64_t x = hash_seed;
  for (uint32_t& k : hash_key_) {
    x += 0x9e3779b97f4a7c15ULL;
    k = static_cast<uint32_t>(mix64(x));
  }

  nodes_.assign(cap, Node{});
  tokens_.resize(cap);

  free_nodes_.reserve(cap);
  // Reverse order so the first node handed out is 1 (0 is the root).
  for (NodeId id = capacity_ - 1; id >= 1; --id) free_nodes_.push_back(id);

  // Power of two so "hash % size" is a mask. <= 50% load keeps linear-probe
  // chains short and guarantees an empty slot, so lookups always end.
  size_t slots = 16;
  while (slots < 2 * cap) slots <<= 1;
  table_.assign(slots, Slot{});
  mask_ = slots - 1;

  heap_.reserve(cap);
  pinned_.reserve(cap);
  block_cached_.assign(static_cast<size_t>(bm.num_blocks()), 0);
}

RadixTree::~RadixTree() {
  for (NodeId id = 1; id < capacity_; ++id) {
    const BlockId b = nodes_[static_cast<size_t>(id)].block;
    if (b != kNoBlock) (void)bm_.decref(b);
  }
}

uint64_t RadixTree::hash_chunk(NodeId parent, const int32_t* tokens) const noexcept {
  // NH hash (from UMAC): sum of (t[2i] + k[2i]) * (t[2i+1] + k[2i+1]).
  //
  // Why not mix the tokens in one at a time: each round waits on the one
  // before it. That version measured 52 ns per block. These 8 multiplies
  // don't depend on each other, so the CPU runs them side by side: 8.5 ns.
  //
  // Why random keys: with fixed keys, a tenant could craft prompts that all
  // hash to the same slot and turn every lookup into a long scan for
  // everyone. With random keys (picked at startup), NH collisions are
  // provably rare (about 1 in 2^32) no matter what the input is.
  //
  // A collision is still safe: find_slot always memcmp's the real tokens.
  uint64_t acc = 0;
  for (int i = 0; i < kBlockSize; i += 2) {
    const uint32_t a = static_cast<uint32_t>(tokens[i]) + hash_key_[i];
    const uint32_t b = static_cast<uint32_t>(tokens[i + 1]) + hash_key_[i + 1];
    acc += static_cast<uint64_t>(a) * b;
  }
  // The parent id is part of the key. The same 16 tokens at a different
  // position in the prompt are a different prefix with different KV.
  return mix64(acc ^ (static_cast<uint64_t>(static_cast<uint32_t>(parent)) << 32));
}

int64_t RadixTree::find_slot(NodeId parent, const int32_t* tokens, uint64_t h) const noexcept {
  for (size_t i = h & mask_;; i = (i + 1) & mask_) {
    const Slot& s = table_[i];
    if (s.node == kNoNode) return -1;
    // Hash match first (cheap), then the real check. A hash collision alone
    // must never return a block, or we'd serve the wrong KV.
    if (s.hash == h && nodes_[static_cast<size_t>(s.node)].parent == parent &&
        std::memcmp(tokens_[static_cast<size_t>(s.node)].t, tokens, sizeof(TokenChunk)) == 0) {
      return static_cast<int64_t>(i);
    }
  }
}

void RadixTree::table_insert(uint64_t h, NodeId node) noexcept {
  size_t i = h & mask_;
  while (table_[i].node != kNoNode) i = (i + 1) & mask_;
  table_[i] = Slot{h, node};
}

void RadixTree::table_erase(size_t slot) noexcept {
  // Backward-shift delete. Why not tombstones: tombstones pile up with every
  // eviction, probe chains get longer forever, and eventually you need a
  // rehash (an allocation). Shifting entries back keeps the table clean.
  size_t hole = slot;
  size_t j = slot;
  for (;;) {
    j = (j + 1) & mask_;
    if (table_[j].node == kNoNode) break;
    const size_t home = table_[j].hash & mask_;
    // Entry j may move into the hole only if its home slot is not in the
    // cyclic range (hole, j]. Otherwise moving it would put it before its
    // home, and lookups would never find it.
    const bool home_in_range = (hole <= j) ? (hole < home && home <= j) : (hole < home || home <= j);
    if (!home_in_range) {
      table_[hole] = table_[j];
      hole = j;
    }
  }
  table_[hole] = Slot{};
}

int64_t RadixTree::match_prefix(const int32_t* tokens, int64_t n,
                                std::vector<BlockId>& out) noexcept {
  out.clear();
  if (n <= 0 || tokens == nullptr) return 0;

  const uint64_t now = ++clock_;
  nodes_[kRoot].last_access = now;

  NodeId cur = kRoot;
  const int64_t full = n / kBlockSize;
  for (int64_t b = 0; b < full; ++b) {
    const int32_t* chunk = tokens + b * kBlockSize;
    const int64_t slot = find_slot(cur, chunk, hash_chunk(cur, chunk));
    if (slot < 0) break;
    const NodeId child = table_[static_cast<size_t>(slot)].node;
    Node& c = nodes_[static_cast<size_t>(child)];
    // Can't fail while the tree holds its own ref. Stop instead of
    // returning a block the caller doesn't own a ref on.
    if (bm_.incref(c.block) != Status::kOk) break;
    out.push_back(c.block);
    c.last_access = now;
    cur = child;
  }

  // Only leaves are in the heap. Their key just went up, so sift down.
  // Internal nodes on the path just had last_access set above.
  const int32_t pos = nodes_[static_cast<size_t>(cur)].heap_pos;
  if (pos >= 0) heap_sift_down(static_cast<size_t>(pos));

  return static_cast<int64_t>(out.size()) * kBlockSize;
}

Status RadixTree::insert(const int32_t* tokens, int64_t n, const BlockId* blocks,
                         int64_t num_blocks, int32_t* added) noexcept {
  if (added != nullptr) *added = 0;
  if (n < 0 || num_blocks < 0 || (n > 0 && tokens == nullptr) ||
      (num_blocks > 0 && blocks == nullptr)) {
    return Status::kInvalidArgument;
  }
  const int64_t full = n / kBlockSize;
  if (num_blocks < full) return Status::kInvalidArgument;

  // Pass 1 (read-only): walk the part that's already cached.
  NodeId cur = kRoot;
  int64_t d = 0;
  for (; d < full; ++d) {
    const int32_t* chunk = tokens + d * kBlockSize;
    const int64_t slot = find_slot(cur, chunk, hash_chunk(cur, chunk));
    if (slot < 0) break;
    cur = table_[static_cast<size_t>(slot)].node;
  }
  const int64_t to_add = full - d;

  // Can't happen: every node owns a distinct block and there are
  // num_blocks + 1 nodes. Checked anyway so a broken invariant fails
  // loudly instead of popping an empty vector.
  if (to_add > static_cast<int64_t>(free_nodes_.size())) return Status::kOutOfNodes;

  // Pass 2 (validate): check every new block before changing anything.
  // Marking as we go also catches the same id twice in one call.
  for (int64_t i = d; i < full; ++i) {
    const BlockId id = blocks[i];
    Status s = Status::kOk;
    if (!bm_.valid(id)) {
      s = Status::kInvalidBlockId;
    } else if (bm_.ref_count(id) == 0) {
      s = Status::kBlockNotAllocated;  // caller must own a ref
    } else if (bm_.ref_count(id) == std::numeric_limits<int32_t>::max()) {
      s = Status::kRefCountOverflow;
    } else if (block_cached_[static_cast<size_t>(id)] != 0) {
      s = Status::kBlockAlreadyCached;  // one block, one place in the tree
    }
    if (s != Status::kOk) {
      for (int64_t j = d; j < i; ++j) block_cached_[static_cast<size_t>(blocks[j])] = 0;
      return s;
    }
    block_cached_[static_cast<size_t>(id)] = 1;
  }

  // Pass 3 (commit). Nothing below can fail.
  const uint64_t now = ++clock_;
  Node& last = nodes_[static_cast<size_t>(cur)];
  // `cur` is about to get a child, so it stops being a leaf.
  if (to_add > 0 && last.heap_pos >= 0) heap_remove(cur);
  for (NodeId x = cur; x != kNoNode; x = nodes_[static_cast<size_t>(x)].parent) {
    nodes_[static_cast<size_t>(x)].last_access = now;
  }
  if (to_add == 0) {
    if (last.heap_pos >= 0) heap_sift_down(static_cast<size_t>(last.heap_pos));
    return Status::kOk;
  }

  for (int64_t i = d; i < full; ++i) {
    const NodeId child = free_nodes_.back();
    free_nodes_.pop_back();
    const int32_t* chunk = tokens + i * kBlockSize;

    Node& c = nodes_[static_cast<size_t>(child)];
    c.parent = cur;
    c.block = blocks[i];
    c.num_children = 0;
    c.heap_pos = -1;
    c.last_access = now;
    std::memcpy(tokens_[static_cast<size_t>(child)].t, chunk, sizeof(TokenChunk));
    table_insert(hash_chunk(cur, chunk), child);
    ++nodes_[static_cast<size_t>(cur)].num_children;
    (void)bm_.incref(c.block);  // checked in pass 2
    cur = child;
  }
  heap_push(cur);  // only the deepest new node is a leaf

  if (added != nullptr) *added = static_cast<int32_t>(to_add);
  return Status::kOk;
}

int32_t RadixTree::evict(int32_t want) noexcept {
  int32_t freed = 0;
  pinned_.clear();

  while (freed < want && !heap_.empty()) {
    const NodeId leaf = heap_.front();
    heap_remove(leaf);
    // A running request still holds this block. Set it aside and put it
    // back after.
    //
    // Trade-off: if many old leaves are pinned we pop and re-push each one,
    // O(p log n) for p pinned leaves. Tracking "pinned" per node would avoid
    // that, but it's a second ref count that can drift from the real one.
    // Eviction only runs when the pool is empty, so we pay the cost there.
    if (bm_.ref_count(nodes_[static_cast<size_t>(leaf)].block) > 1) {
      pinned_.push_back(leaf);  // capacity reserved: no realloc
      continue;
    }
    remove_leaf(leaf);
    ++freed;
  }

  for (const NodeId id : pinned_) heap_push(id);
  pinned_.clear();
  return freed;
}

void RadixTree::remove_leaf(NodeId leaf) noexcept {
  Node& n = nodes_[static_cast<size_t>(leaf)];
  const NodeId parent = n.parent;
  const int32_t* chunk = tokens_[static_cast<size_t>(leaf)].t;

  const int64_t slot = find_slot(parent, chunk, hash_chunk(parent, chunk));
  if (slot >= 0) table_erase(static_cast<size_t>(slot));

  block_cached_[static_cast<size_t>(n.block)] = 0;
  (void)bm_.decref(n.block);  // ref was 1, so the block goes back to the pool

  n = Node{};
  free_nodes_.push_back(leaf);

  Node& p = nodes_[static_cast<size_t>(parent)];
  --p.num_children;
  // The parent may be a leaf now. It can be evicted in this same call.
  if (parent != kRoot && p.num_children == 0) heap_push(parent);
}

bool RadixTree::lru_before(NodeId a, NodeId b) const noexcept {
  const uint64_t ta = nodes_[static_cast<size_t>(a)].last_access;
  const uint64_t tb = nodes_[static_cast<size_t>(b)].last_access;
  // Tie-break on id so eviction order is deterministic.
  return ta < tb || (ta == tb && a < b);
}

void RadixTree::heap_place(size_t pos, NodeId id) noexcept {
  heap_[pos] = id;
  nodes_[static_cast<size_t>(id)].heap_pos = static_cast<int32_t>(pos);
}

void RadixTree::heap_sift_up(size_t pos) noexcept {
  const NodeId id = heap_[pos];
  while (pos > 0) {
    const size_t parent = (pos - 1) / 2;
    if (!lru_before(id, heap_[parent])) break;
    heap_place(pos, heap_[parent]);
    pos = parent;
  }
  heap_place(pos, id);
}

void RadixTree::heap_sift_down(size_t pos) noexcept {
  const NodeId id = heap_[pos];
  const size_t size = heap_.size();
  for (;;) {
    size_t child = 2 * pos + 1;
    if (child >= size) break;
    if (child + 1 < size && lru_before(heap_[child + 1], heap_[child])) ++child;
    if (!lru_before(heap_[child], id)) break;
    heap_place(pos, heap_[child]);
    pos = child;
  }
  heap_place(pos, id);
}

void RadixTree::heap_push(NodeId id) noexcept {
  heap_.push_back(id);  // capacity reserved: no realloc
  heap_sift_up(heap_.size() - 1);
}

void RadixTree::heap_remove(NodeId id) noexcept {
  const auto pos = static_cast<size_t>(nodes_[static_cast<size_t>(id)].heap_pos);
  nodes_[static_cast<size_t>(id)].heap_pos = -1;
  const NodeId moved = heap_.back();
  heap_.pop_back();
  if (pos == heap_.size()) return;  // removed the last element
  heap_place(pos, moved);
  heap_sift_down(pos);
  heap_sift_up(static_cast<size_t>(nodes_[static_cast<size_t>(moved)].heap_pos));
}

int32_t RadixTree::num_cached_blocks() const noexcept {
  return capacity_ - 1 - static_cast<int32_t>(free_nodes_.size());
}

bool RadixTree::is_cached(BlockId id) const noexcept {
  return bm_.valid(id) && block_cached_[static_cast<size_t>(id)] != 0;
}

bool RadixTree::check_invariants(std::string* why) const {
  auto fail = [why](std::string msg) {
    if (why != nullptr) *why = std::move(msg);
    return false;
  };
  auto s = [](auto v) { return std::to_string(v); };

  const auto cap = static_cast<size_t>(capacity_);
  std::vector<int32_t> kids(cap, 0);
  int32_t used = 0;
  int32_t leaves = 0;

  for (NodeId id = 1; id < capacity_; ++id) {
    const Node& n = nodes_[static_cast<size_t>(id)];
    if (n.block == kNoBlock) {
      if (n.parent != kNoNode || n.num_children != 0 || n.heap_pos != -1) {
        return fail("free node " + s(id) + " has stale fields");
      }
      continue;
    }
    ++used;
    if (n.parent < 0 || n.parent >= capacity_) return fail("node " + s(id) + " has bad parent");
    const Node& p = nodes_[static_cast<size_t>(n.parent)];
    if (n.parent != kRoot && p.block == kNoBlock) return fail("node " + s(id) + " has a free parent");
    ++kids[static_cast<size_t>(n.parent)];
    if (!bm_.valid(n.block)) return fail("node " + s(id) + " has invalid block");
    if (bm_.ref_count(n.block) < 1) return fail("node " + s(id) + " holds a free block");
    if (block_cached_[static_cast<size_t>(n.block)] == 0) return fail("block " + s(n.block) + " not marked cached");
    if (n.last_access > p.last_access) return fail("node " + s(id) + " newer than its parent");
    const int32_t* t = tokens_[static_cast<size_t>(id)].t;
    const int64_t slot = find_slot(n.parent, t, hash_chunk(n.parent, t));
    if (slot < 0 || table_[static_cast<size_t>(slot)].node != id) {
      return fail("node " + s(id) + " not findable in hash table");
    }
  }

  int32_t marked = 0;
  for (const uint8_t m : block_cached_) marked += (m != 0) ? 1 : 0;
  if (marked != used) return fail("block_cached_ count " + s(marked) + " != nodes " + s(used));

  if (static_cast<size_t>(used) + free_nodes_.size() != cap - 1) {
    return fail("used + free nodes != capacity");
  }

  for (NodeId id = 0; id < capacity_; ++id) {
    const Node& n = nodes_[static_cast<size_t>(id)];
    if (kids[static_cast<size_t>(id)] != n.num_children) {
      return fail("node " + s(id) + " num_children is wrong");
    }
    const bool is_leaf = id != kRoot && n.block != kNoBlock && n.num_children == 0;
    if (is_leaf) ++leaves;
    if (is_leaf != (n.heap_pos >= 0)) {
      return fail("node " + s(id) + (is_leaf ? " is a leaf but not in heap" : " is in heap but not a leaf"));
    }
  }

  if (static_cast<size_t>(leaves) != heap_.size()) return fail("heap size != leaf count");
  for (size_t i = 0; i < heap_.size(); ++i) {
    if (nodes_[static_cast<size_t>(heap_[i])].heap_pos != static_cast<int32_t>(i)) {
      return fail("heap_pos mismatch at " + s(i));
    }
    if (i > 0 && lru_before(heap_[i], heap_[(i - 1) / 2])) return fail("heap order broken at " + s(i));
  }

  int32_t occupied = 0;
  for (const Slot& sl : table_) occupied += (sl.node != kNoNode) ? 1 : 0;
  if (occupied != used) return fail("hash table holds " + s(occupied) + " entries, expected " + s(used));

  return true;
}

}  // namespace prefixserve
