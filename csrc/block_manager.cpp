#include "block_manager.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace prefixserve {

const char* to_string(Status s) noexcept {
  switch (s) {
    case Status::kOk:
      return "ok";
    case Status::kInvalidBlockId:
      return "block id out of range";
    case Status::kBlockNotAllocated:
      return "block is not allocated (double free or use-after-free)";
    case Status::kRefCountOverflow:
      return "ref count overflow";
    case Status::kBlockAlreadyCached:
      return "block is already owned by the radix tree";
    case Status::kOutOfNodes:
      return "radix tree node pool exhausted";
    case Status::kInvalidArgument:
      return "invalid argument";
  }
  return "unknown status";
}

BlockManager::BlockManager(int32_t num_blocks) : num_blocks_(num_blocks) {
  if (num_blocks <= 0 || num_blocks > kMaxBlocks) {
    throw std::invalid_argument("num_blocks must be in [1, " +
                                std::to_string(kMaxBlocks) + "], got " +
                                std::to_string(num_blocks));
  }
  const auto n = static_cast<size_t>(num_blocks);
  ref_count_.assign(n, 0);
  // Full capacity now, so decref's push_back can never reallocate later.
  free_stack_.reserve(n);
  // Pushed in reverse so the first allocate() returns block 0.
  for (BlockId id = num_blocks - 1; id >= 0; --id) free_stack_.push_back(id);
}

bool BlockManager::check_invariants(std::string* why) const {
  auto fail = [why](std::string msg) {
    if (why != nullptr) *why = std::move(msg);
    return false;
  };

  std::vector<uint8_t> on_free_list(static_cast<size_t>(num_blocks_), 0);
  for (const BlockId id : free_stack_) {
    if (!valid(id)) return fail("free list holds invalid id " + std::to_string(id));
    auto& seen = on_free_list[static_cast<size_t>(id)];
    if (seen != 0) return fail("block " + std::to_string(id) + " is on the free list twice");
    seen = 1;
  }
  for (BlockId id = 0; id < num_blocks_; ++id) {
    const int32_t rc = ref_count_[static_cast<size_t>(id)];
    const bool is_free = on_free_list[static_cast<size_t>(id)] != 0;
    if (rc < 0) return fail("block " + std::to_string(id) + " has negative ref_count");
    if ((rc == 0) != is_free) {
      return fail("block " + std::to_string(id) + " has ref_count " + std::to_string(rc) +
                  (is_free ? " but is on the free list" : " but is not on the free list"));
    }
  }
  return true;
}

}  // namespace prefixserve
