#include "system/HoppingBlocks.hpp"

namespace cpb {

idx_t HoppingBlocks::nnz() const {
    return std::accumulate(blocks.begin(), blocks.end(), idx_t{0}, [](idx_t n, Block const& b) {
        return n + static_cast<idx_t>(b.size());
    });
}

void HoppingBlocks::reserve(ArrayXi const& counts) {
    assert(counts.size() == static_cast<idx_t>(blocks.size()));
    for (auto i = size_t{0}; i < blocks.size(); ++i) {
        blocks[i].reserve(counts[i]);
    }
}

void HoppingBlocks::append(HopID family_id, ArrayXi&& rows, ArrayXi&& cols) {
    assert(rows.size() == cols.size());
    auto& block = blocks[family_id.as<size_t>()];
    block.resize(rows.size());

    for (auto i = 0; i < rows.size(); ++i) {
        auto m = rows[i];
        auto n = cols[i];
        if (m > n) { std::swap(m, n); } // upper triangular format
        block[i] = {m, n};
    }

    // Maintain upper triangular format
    std::sort(block.begin(), block.end());
    block.erase(std::unique(block.begin(), block.end()), block.end());
}

SparseMatrixX<storage_idx_t> HoppingBlocks::to_csr() const {
    auto csr = SparseMatrixX<storage_idx_t>(num_sites, num_sites);
    csr.reserve(nnz());
    for (auto const& block : *this) {
        for (auto const& coo : block.coordinates()) {
            csr.insert(coo.row, coo.col) = block.family_id().value();
        }
    }
    csr.makeCompressed();
    return csr.markAsRValue();
}

} // namespace cpb
