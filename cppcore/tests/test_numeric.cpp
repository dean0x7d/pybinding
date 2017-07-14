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
    REQUIRE_THROWS_WITH(RealArrayRef{arrayref(v2)}, "Invalid VariantArrayRef assignment");
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
    REQUIRE_THROWS_WITH((num::match2sp<VectorX, VectorX>(ref1, ref2, ArrayRefTestOp2{})),
                        "A match was not found");
}

TEST_CASE("Aligned size") {
    REQUIRE((num::aligned_size<float, 16>(4) == 4));
    REQUIRE((num::aligned_size<std::complex<double>, 16>(2) == 2));
    REQUIRE((num::aligned_size<std::complex<float>, 32>(9) == 12));
}

TEST_CASE("concat") {
    auto const x1 = ArrayXf::Constant(3, 1).eval();
    auto const x2 = ArrayXf::LinSpaced(3, 2, 4).eval();
    auto expected_x = ArrayXf(6);
    expected_x << 1, 1, 1, 2, 3, 4;

    auto const result_x = concat(x1, x2);
    REQUIRE(result_x.isApprox(expected_x));

    auto const y1 = ArrayXf::Constant(3, 2).eval();
    auto const y2 = ArrayXf::LinSpaced(3, 3, 5).eval();
    auto expected_y = ArrayXf(6);
    expected_y << 2, 2, 2, 3, 4, 5;

    auto const result_y = concat(y1, y2);
    REQUIRE(result_y.isApprox(expected_y));

    auto const z1 = ArrayXf::Constant(3, 0).eval();
    auto const z2 = ArrayXf::Constant(3, -1).eval();
    auto expected_z = ArrayXf(6);
    expected_z << 0, 0, 0, -1, -1, -1;

    auto const result_z = concat(z1, z2);
    REQUIRE(result_z.isApprox(expected_z));

    auto const r1 = CartesianArray(x1, y1, z1);
    auto const r2 = CartesianArray(x2, y2, z2);
    auto const result = concat(r1, r2);
    REQUIRE(result.x.isApprox(expected_x));
    REQUIRE(result.y.isApprox(expected_y));
    REQUIRE(result.z.isApprox(expected_z));
}
