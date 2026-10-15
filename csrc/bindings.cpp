// Python bindings: prefixserve._core ("PrefixServeCore").
//
// Memory rule at this boundary:
//   - Inputs: we borrow the numpy buffer only for the length of the call.
//     We never keep the pointer, so Python can free it afterwards.
//   - Outputs: always a fresh Python list (a copy). Never a numpy view of
//     C++ memory. A view outlives the call, the next match/evict reuses that
//     memory, and Python reads garbage or segfaults. The copy is ~100 ints,
//     well under a microsecond.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "block_manager.hpp"
#include "radix_tree.hpp"

namespace py = pybind11;
namespace ps = prefixserve;

namespace {

// Broken ref counting or a bad block id. Always a scheduler bug, so loud.
class CoreError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

void check(ps::Status s, const char* what) {
  if (s != ps::Status::kOk) throw CoreError(std::string(what) + ": " + ps::to_string(s));
}

// forcecast lets callers pass a plain list too. numpy converts it to int32
// on the Python side before we see it.
using IntArray = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

void require_1d(const IntArray& a, const char* name) {
  if (a.ndim() != 1) throw py::value_error(std::string(name) + " must be 1-D");
}

py::list to_list(const std::vector<ps::BlockId>& v) {
  py::list out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = py::int_(v[i]);
  return out;
}

// Owns the tree plus a reusable output buffer, so match_prefix never
// allocates on the C++ side.
class PyRadixTree {
 public:
  explicit PyRadixTree(ps::BlockManager& bm) : tree_(bm) {
    scratch_.reserve(static_cast<size_t>(bm.num_blocks()));
  }

  py::tuple match_prefix(const IntArray& tokens) {
    require_1d(tokens, "tokens");
    const int64_t matched = tree_.match_prefix(tokens.data(), tokens.size(), scratch_);
    return py::make_tuple(matched, to_list(scratch_));
  }

  int32_t insert(const IntArray& tokens, const IntArray& block_ids) {
    require_1d(tokens, "tokens");
    require_1d(block_ids, "block_ids");
    int32_t added = 0;
    check(tree_.insert(tokens.data(), tokens.size(), block_ids.data(), block_ids.size(), &added),
          "insert");
    return added;
  }

  int32_t evict(int32_t num_blocks) {
    if (num_blocks < 0) throw py::value_error("num_blocks must be >= 0");
    return tree_.evict(num_blocks);
  }

  int32_t num_cached_blocks() const { return tree_.num_cached_blocks(); }
  bool is_cached(ps::BlockId id) const { return tree_.is_cached(id); }

  void check_invariants() const {
    std::string why;
    if (!tree_.check_invariants(&why)) throw CoreError("RadixTree invariant broken: " + why);
  }

 private:
  ps::RadixTree tree_;
  std::vector<ps::BlockId> scratch_;
};

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "PrefixServeCore: paged KV block manager and block-granular radix prefix cache.";
  m.attr("BLOCK_SIZE") = ps::kBlockSize;

  py::register_exception<CoreError>(m, "CoreError", PyExc_RuntimeError);

  // Not releasing the GIL anywhere. Every call here takes microseconds;
  // dropping and re-taking the GIL would cost about as much as the call.
  py::class_<ps::BlockManager>(m, "BlockManager")
      .def(py::init<int32_t>(), py::arg("num_blocks"))
      .def("allocate",
           [](ps::BlockManager& bm) -> std::optional<ps::BlockId> { return bm.allocate(); },
           "One block with ref_count 1, or None when the pool is empty.")
      .def(
          "allocate_many",
          [](ps::BlockManager& bm, int32_t n) -> py::object {
            if (n < 0) throw py::value_error("n must be >= 0");
            // All-or-nothing. Half an allocation would leak blocks if the
            // caller bails out on None.
            if (bm.num_free() < n) return py::none();
            py::list out(static_cast<size_t>(n));
            for (int32_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = py::int_(*bm.allocate());
            return std::move(out);
          },
          py::arg("n"), "n blocks, or None (and nothing allocated) if fewer than n are free.")
      .def(
          "incref", [](ps::BlockManager& bm, ps::BlockId id) { check(bm.incref(id), "incref"); },
          py::arg("block_id"))
      .def(
          "decref", [](ps::BlockManager& bm, ps::BlockId id) { check(bm.decref(id), "decref"); },
          py::arg("block_id"))
      .def(
          "decref_many",
          [](ps::BlockManager& bm, const IntArray& ids) {
            require_1d(ids, "block_ids");
            // One Python call instead of one per block. Stops at the first
            // error: that's a scheduler bug, not something to recover from.
            const int32_t* p = ids.data();
            for (py::ssize_t i = 0; i < ids.size(); ++i) {
              const ps::Status s = bm.decref(p[i]);
              if (s != ps::Status::kOk) {
                throw CoreError("decref_many: block " + std::to_string(p[i]) + " at index " +
                                std::to_string(i) + ": " + ps::to_string(s));
              }
            }
          },
          py::arg("block_ids"))
      .def(
          "ref_count",
          [](const ps::BlockManager& bm, ps::BlockId id) {
            if (!bm.valid(id)) check(ps::Status::kInvalidBlockId, "ref_count");
            return bm.ref_count(id);
          },
          py::arg("block_id"))
      .def_property_readonly("num_free", &ps::BlockManager::num_free)
      .def_property_readonly("num_blocks", &ps::BlockManager::num_blocks)
      .def("check_invariants", [](const ps::BlockManager& bm) {
        std::string why;
        if (!bm.check_invariants(&why)) throw CoreError("BlockManager invariant broken: " + why);
      });

  py::class_<PyRadixTree>(m, "RadixTree")
      // keep_alive<1, 2>: the tree keeps the BlockManager's Python object
      // alive. Without it, `del bm` would free the C++ BlockManager while
      // the tree still points at it.
      .def(py::init<ps::BlockManager&>(), py::arg("block_manager"), py::keep_alive<1, 2>())
      .def("match_prefix", &PyRadixTree::match_prefix, py::arg("tokens"),
           "(matched_tokens, block_ids). Takes 1 ref per block for the caller.")
      .def("insert", &PyRadixTree::insert, py::arg("tokens"), py::arg("block_ids"),
           "Caches full blocks. Returns how many new blocks the tree now holds.")
      .def("evict", &PyRadixTree::evict, py::arg("num_blocks"),
           "Frees up to num_blocks unused blocks, LRU first. Returns how many were freed.")
      .def("is_cached", &PyRadixTree::is_cached, py::arg("block_id"))
      .def_property_readonly("num_cached_blocks", &PyRadixTree::num_cached_blocks)
      .def("check_invariants", &PyRadixTree::check_invariants);
}
