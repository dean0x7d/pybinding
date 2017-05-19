#include "system/HoppingBlocks.hpp"

namespace cpb {

HoppingBlocks::HoppingBlocks(idx_t num_sites, SerializedBlocks const& data, NameMap name_map)
    : num_sites(num_sites), name_map(std::move(name_map)) {
    blocks.reserve(data.size());
    for (auto const& pair : data) {
        auto const size = pair.first.size();
        auto block = std::vector<COO>(static_cast<size_t>(size));
        for (auto i = 0; i < size; ++i) {
            block[i].row = pair.first[i];
            block[i].col = pair.second[i];
        }
        blocks.push_back(std::move(block));
    }
}

HoppingBlocks::SerializedBlocks HoppingBlocks::get_serialized_blocks() const {
    auto data = SerializedBlocks();
    for (auto const& block : *this) {
        auto row = ArrayXi(block.size());
        auto col = ArrayXi(block.size());
        auto n = 0;
        for (auto const& coo : block.coordinates()) {
            row[n] = coo.row;
            col[n] = coo.col;
            ++n;
        }
        data.emplace_back(std::move(row), std::move(col));
    }
    return data;
}

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
    block.reserve(block.size() + rows.size());

    for (auto i = 0; i < rows.size(); ++i) {
        auto m = rows[i];
        auto n = cols[i];
        if (m > n) { std::swap(m, n); } // upper triangular format
        block.emplace_back(m, n);
    }

    // Maintain upper triangular format
    std::sort(block.begin(), block.end());
    block.erase(std::unique(block.begin(), block.end()), block.end());
}

SparseMatrixX<storage_idx_t> HoppingBlocks::tocsr() const {
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
