#include <catch.hpp>

#include "compute/lanczos.hpp"
#include "compute/kernel_polynomial.hpp"
#include "fixtures.hpp"
using namespace cpb;

TEST_CASE("Lanczos", "[lanczos]") {
    auto const model = Model(graphene::monolayer(), Primitive(5, 5),
                             TranslationalSymmetry(1, 1));

    auto const& matrix = ham::get_reference<std::complex<float>>(model.hamiltonian());
    auto loop_counters = std::vector<int>(3);
    for (auto& count : loop_counters) {
        auto const bounds = compute::minmax_eigenvalues(matrix, 1e-3f);
        auto const expected = abs(3 * graphene::t);
        REQUIRE(bounds.max == Approx(expected));
        REQUIRE(bounds.min == Approx(-expected));
        count = bounds.loops;
    }

    auto const all_equal = std::all_of(loop_counters.begin(), loop_counters.end(),
                                       [&](int c) { return c == loop_counters.front(); });
    REQUIRE(all_equal);
}

template<class scalar_t>
SparseMatrixX<scalar_t> make_random_csr(idx_t rows, idx_t cols) {
    using real_t = num::get_real_t<scalar_t>;
    using complex_t = num::get_complex_t<scalar_t>;

    auto generator = std::mt19937();
    auto distribution = std::uniform_real_distribution<real_t>(0.0, 1.0);

    auto triplets = std::vector<Eigen::Triplet<scalar_t>>();
    for (auto i = storage_idx_t{0}; i < rows; ++i) {
        for (auto j = storage_idx_t{0}; j < cols; ++j) {
            auto value = static_cast<scalar_t>(distribution(generator));
            var::variant<real_t*, complex_t*>(&value).match(
                [](real_t*) { },
                [](complex_t* p) { p->imag(real_t{0.5} * p->real()); }
            );
            if (abs(value) < 0.1) { triplets.emplace_back(i, j, value); }
        }
    }

    auto m = SparseMatrixX<scalar_t>(rows, cols);
    m.setFromTriplets(triplets.begin(), triplets.end());
    m.makeCompressed();
    return m.markAsRValue();
}

template<class real_t>
bool approx_equal(real_t a, real_t b) {
    return a == Approx(b);
}

template<class real_t>
bool approx_equal(std::complex<real_t> a, std::complex<real_t> b) {
    return a.real() == Approx(b.real()) && a.imag() == Approx(b.imag());
}

template<class scalar_t, size_t size>
bool approx_equal(std::array<scalar_t, size> const& a, std::array<scalar_t, size> const& b) {
    auto ma = Eigen::Map<ArrayX<scalar_t> const>(a.data(), size);
    auto mb = Eigen::Map<ArrayX<scalar_t> const>(b.data(), size);
    return ma.isApprox(mb);
};

template<class scalar_t>
SparseMatrixX<scalar_t> convert_sparse(SparseMatrixX<scalar_t> const& m,
                                       var::tag<SparseMatrixX<scalar_t>>) {
    return m;
}

template<class scalar_t>
num::EllMatrix<scalar_t> convert_sparse(SparseMatrixX<scalar_t> const& m,
                                        var::tag<num::EllMatrix<scalar_t>>) {
    return num::csr_to_ell(m);
}

template<class SparseMatrix, class scalar_t = typename SparseMatrix::Scalar>
void test_kpm_spmv(idx_t size) {
    constexpr auto cols = static_cast<idx_t>(simd::traits<scalar_t>::size);

    auto const ref_matrix = make_random_csr<scalar_t>(size, size);
    auto const matrix = convert_sparse(ref_matrix, var::tag<SparseMatrix>{});
    auto const x = VectorX<scalar_t>::Random(size).eval();
    auto const y = VectorX<scalar_t>::Random(size).eval();
    auto const xx = MatrixX<scalar_t>::Random(size, cols).eval();
    auto const yy = MatrixX<scalar_t>::Random(size, cols).eval();

    auto const expected_r = (ref_matrix * x - y).eval();
    auto const expected_m2 = static_cast<scalar_t>(x.squaredNorm());
    auto const expected_m3 = static_cast<scalar_t>(expected_r.dot(x));

    auto const expected_rr = (ref_matrix * xx - yy).eval();
    auto const expected_m22 = [&] {
        auto m = simd::array<scalar_t>{{0}};
        for (auto i = 0; i < cols; ++i) {
            m[i] = xx.col(i).squaredNorm();
        }
        return m;
    }();
    auto const expected_m33 = [&] {
        auto m = simd::array<scalar_t>{{0}};
        for (auto i = 0; i < cols; ++i) {
            m[i] = expected_rr.col(i).dot(xx.col(i));
        }
        return m;
    }();

    auto r = VectorX<scalar_t>();
    auto rr = MatrixX<scalar_t>();
    auto m2 = scalar_t{0};
    auto m3 = scalar_t{0};
    auto m22 = simd::array<scalar_t>{{0}};
    auto m33 = simd::array<scalar_t>{{0}};

    auto reset_variables = [&]() {
        r = y;
        rr = yy;
        m2 = scalar_t{0};
        m3 = scalar_t{0};
        m22 = simd::array<scalar_t>{{0}};
        m33 = simd::array<scalar_t>{{0}};
    };

    reset_variables();
    compute::kpm_spmv(0, size, matrix, x, r);
    REQUIRE(r.isApprox(expected_r));

    reset_variables();
    compute::kpm_spmv(0, size, matrix, xx, rr);
    REQUIRE(rr.isApprox(expected_rr));

    reset_variables();
    compute::kpm_spmv_diagonal(0, size, matrix, x, r, m2, m3);
    REQUIRE(r.isApprox(expected_r));
    REQUIRE(approx_equal(m2, expected_m2));
    REQUIRE(approx_equal(m3, expected_m3));

    reset_variables();
    compute::kpm_spmv_diagonal(0, size, matrix, xx, rr, m22, m33);
    REQUIRE(rr.isApprox(expected_rr));
    REQUIRE(approx_equal(m22, expected_m22));
    REQUIRE(approx_equal(m33, expected_m33));
}

TEST_CASE("KPM SpMV") {
    constexpr auto size = 100;

    test_kpm_spmv<SparseMatrixX<float>>(size);
    test_kpm_spmv<SparseMatrixX<std::complex<float>>>(size);
    test_kpm_spmv<SparseMatrixX<double>>(size);
    test_kpm_spmv<SparseMatrixX<std::complex<double>>>(size);

    test_kpm_spmv<num::EllMatrix<float>>(size);
    test_kpm_spmv<num::EllMatrix<std::complex<float>>>(size);
    test_kpm_spmv<num::EllMatrix<double>>(size);
    test_kpm_spmv<num::EllMatrix<std::complex<double>>>(size);
}
