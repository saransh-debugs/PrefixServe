#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace prefixserve {

using BlockId = int32_t;

// Tokens per KV block.
inline constexpr int32_t kBlockSize = 16;
inline constexpr BlockId kNoBlock = -1;

// Upper bound on pool size. 2^26 blocks = 1B tokens. Keeps every index
// inside int32 with room to spare.
inline constexpr int32_t kMaxBlocks = 1 << 26;

// Errors are return values, not exceptions.
// Why: these calls run every scheduler step. A throw unwinds the stack and
// allocates the exception object, which is slow and breaks the
// "no allocations in the hot path" rule. Only constructors throw.
enum class Status : uint8_t {
  kOk = 0,
  kInvalidBlockId,      // id outside [0, num_blocks)
  kBlockNotAllocated,   // ref op on a free block: double free or use-after-free
  kRefCountOverflow,
  kBlockAlreadyCached,  // block is already owned by a radix tree node
  kOutOfNodes,          // tree node pool empty (means an invariant broke)
  kInvalidArgument,
};

const char* to_string(Status s) noexcept;

// Fixed pool of KV cache blocks with ref counting.
//
// All memory is allocated once, in the constructor. After that nothing here
// allocates: the free list is a vector with capacity == num_blocks, and a
// block can only be on it once (decref refuses to go below 0).
//
// There is no free_block(). The only way to free a block is decref to 0.
// A direct free would ignore other owners (the radix tree, other requests
// sharing the prefix) and hand their memory to someone else.
//
// Not thread-safe, on purpose. One asyncio task owns the scheduler. With no
// second thread, a mutex or lock-free free list would only add cost.
class BlockManager {
 public:
  explicit BlockManager(int32_t num_blocks);

  BlockManager(const BlockManager&) = delete;
  BlockManager& operator=(const BlockManager&) = delete;
  BlockManager(BlockManager&&) = delete;
  BlockManager& operator=(BlockManager&&) = delete;
  ~BlockManager() = default;

  // Hot-path methods are defined in the header so they inline into the
  // radix tree and bindings. Out of line, each would be a real call.

  // Returns a block with ref_count 1, or nullopt when the pool is empty.
  // Running out is normal (the scheduler evicts or preempts), so it's not
  // an error.
  [[nodiscard]] std::optional<BlockId> allocate() noexcept {
    if (free_stack_.empty()) return std::nullopt;
    // LIFO: the most recently freed block comes back first. Its ref_count
    // slot is likely still in CPU cache. A FIFO queue would hand out the
    // coldest block every time.
    const BlockId id = free_stack_.back();
    free_stack_.pop_back();
    ref_count_[static_cast<size_t>(id)] = 1;
    return id;
  }

  [[nodiscard]] Status incref(BlockId id) noexcept {
    if (!valid(id)) return Status::kInvalidBlockId;
    int32_t& rc = ref_count_[static_cast<size_t>(id)];
    if (rc == 0) return Status::kBlockNotAllocated;
    if (rc == std::numeric_limits<int32_t>::max()) return Status::kRefCountOverflow;
    ++rc;
    return Status::kOk;
  }

  // At 0 the block goes back on the free list.
  [[nodiscard]] Status decref(BlockId id) noexcept {
    if (!valid(id)) return Status::kInvalidBlockId;
    int32_t& rc = ref_count_[static_cast<size_t>(id)];
    // Always checked, not just in debug builds. A double free here would
    // put the block on the free list twice, and two requests would later
    // write their KV into the same memory. That bug is silent and awful.
    if (rc == 0) return Status::kBlockNotAllocated;
    if (--rc == 0) free_stack_.push_back(id);  // capacity reserved: no realloc
    return Status::kOk;
  }

  // 0 for a free block, -1 for an invalid id.
  [[nodiscard]] int32_t ref_count(BlockId id) const noexcept {
    return valid(id) ? ref_count_[static_cast<size_t>(id)] : -1;
  }

  [[nodiscard]] bool valid(BlockId id) const noexcept {
    return id >= 0 && id < num_blocks_;
  }

  [[nodiscard]] int32_t num_free() const noexcept {
    return static_cast<int32_t>(free_stack_.size());
  }

  [[nodiscard]] int32_t num_blocks() const noexcept { return num_blocks_; }

  // O(n) full consistency check. Tests only.
  bool check_invariants(std::string* why) const;

 private:
  int32_t num_blocks_;
  std::vector<int32_t> ref_count_;
  std::vector<BlockId> free_stack_;
};

}  // namespace prefixserve
