#include <catch.hpp>

#include "numeric/dense.hpp"
using namespace cpb;

struct ArrayRefTestOp {
    template<class Vector>
    num::Tag operator()(Vector v) const {
        v[3] = 0;
        return num::detail::get_tag<typename Vector::Scalar>();
    }
};

TEST_CASE("ArrayRef and match", "[arrayref]") {
    Eigen::Vector4d v1(0, 1, 2, 3);
    auto ref1 = RealArrayRef{arrayref(v1)};
    REQUIRE(num::match<VectorX>(ref1, ArrayRefTestOp{}) == num::Tag::f64);
    REQUIRE(v1[3] == .0);

    Eigen::VectorXcf v2(4); v2 << 0, 1, 2, 3;
    REQUIRE_THROWS(RealArrayRef{arrayref(v2)});
    auto ref2 = ComplexArrayRef{arrayref(v2)};
    REQUIRE(num::match<VectorX>(ref2, ArrayRefTestOp{}) == num::Tag::cf32);
    REQUIRE(v2[3] == .0f);
}

struct ArrayRefTestOp2 {
    template<class Vector1, class Vector2>
    num::Tag operator()(Vector1 v1, Vector2 v2) const {
        using Scalar2 = typename Vector2::Scalar;
        v2[3] =v1.template cast<Scalar2>()[0];
        return num::detail::get_tag<Scalar2>();
    }
};

TEST_CASE("ArrayRef and match2", "[arrayref]") {
    Eigen::Vector4f v1(0, 1, 2, 3);
    auto ref1 = RealArrayConstRef{arrayref(v1)};
    Eigen::Vector4cd v2(0, 1, 2, 3);
    auto ref2 = ComplexArrayRef{arrayref(v2)};

    REQUIRE((num::match2<VectorX, VectorX>(ref1, ref2, ArrayRefTestOp2{})) == num::Tag::cf64);
    REQUIRE(v2[3] == .0);
    REQUIRE_THROWS((num::match2sp<VectorX, VectorX>(ref1, ref2, ArrayRefTestOp2{})));
}

TEST_CASE("Aligned size") {
    REQUIRE((num::aligned_size<float, 16>(4) == 4));
    REQUIRE((num::aligned_size<std::complex<double>, 16>(2) == 2));
    REQUIRE((num::aligned_size<std::complex<float>, 32>(9) == 12));
}
