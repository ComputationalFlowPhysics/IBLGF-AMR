#include <gtest/gtest.h>
#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/types.hpp>
#include <iblgf/domain/svt.hpp>
#include <cmath>
#include <array>


namespace iblgf
{
using namespace types;

// Test 2D pure translation (no rotation)
TEST(svt_test, pure_translation_2d)
{
    // clang-format off
    std::string configStr = "                   \
        svt                                     \
        {                                       \
            p = -1.0;                           \
            m = 1.0;                            \
            Omega0 = 0.0;                       \
            U0 = 1.0;                           \
            alpha0 = 0.0;                       \
            x0 = 0.0;                           \
        }                                       \
    ";
    // clang-format on

    dictionary::Dictionary dict("svt", configStr);
    
    ib::SVT<2> svt;
    svt.init(&dict);

    // Verify parameters were loaded correctly
    EXPECT_EQ(svt.p, -1.0);
    EXPECT_EQ(svt.m, 1.0);
    EXPECT_EQ(svt.omega0, 0.0);
    EXPECT_EQ(svt.U0, 1.0);
    EXPECT_EQ(svt.alpha0, 0.0);

    float_type t = 1.0;
    types::vector_type<float_type, 2> coord(std::array<float_type, 2>{0.5, 0.3});

    // For pure translation with alpha0=0: u_t = (U0*t^m, 0)
    // Expected: idx=0 gives -U0*t^m*cos(0) = -1.0
    //           idx=1 gives U0*t^m*sin(0) = 0.0
    float_type u_x = svt(0, t, coord);
    float_type u_y = svt(1, t, coord);

    EXPECT_NEAR(u_x, -1.0, 1e-10);
    EXPECT_NEAR(u_y, 0.0, 1e-10);
}

// Test 2D pure translation at AoA (no rotation)
TEST(svt_test,pure_translation_aoa_2d)
{
    std::string configStr = "                   \
        svt                                     \
        {                                       \
            p = -1.0;                           \
            m = 0.0;                            \
            Omega0 = 0.0;                       \
            U0 = 1.0;                           \
            alpha0 = 0.7853981634;              \
            x0 = 0.0;                           \
        }                                       \
    ";

    dictionary::Dictionary dict("svt", configStr);
    ib::SVT<2> svt;
    svt.init(&dict);
    float_type t = 1.0;
    types::vector_type<float_type, 2> coord(std::array<float_type, 2>{0.5, 0.3});
    // For pure translation with alpha0=pi/4: u_t = (U0*t^m*cos(pi/4), -U0*t^m*sin(pi/4))
    // Expected: idx=0 gives -U0*t^m*cos(pi/4) = -1.0*std::cos(0.7853981634) = -0.7071067812
    //           idx=1 gives U0*t^m*sin(pi /4) = 1.0*std::sin(0.7853981634) = 0.7071067812
    float_type u_x = svt(0, t, coord);
    float_type u_y = svt(1, t, coord);
    EXPECT_NEAR(u_x, -0.7071067812, 1e-10);
    EXPECT_NEAR(u_y, 0.7071067812, 1e-10);


}
// Test 2D pure translation in y (no rotation)
TEST(svt_test,pure_translation_y_2d)
{
    std::string configStr = "                   \
        svt                                     \
        {                                       \
            p = -1.0;                           \
            m = 0.0;                            \
            Omega0 = 0.0;                       \
            U0 = 1.0;                           \
            alpha0 = 1.5707963267948966;        \
            x0 = 0.0;                           \
        }                                       \
    ";

    dictionary::Dictionary dict("svt", configStr);
    ib::SVT<2> svt;
    svt.init(&dict);
    float_type t = 1.0;
    types::vector_type<float_type, 2> coord(std::array<float_type, 2>{0.5, 0.3});
    // For pure translation with alpha0=pi/2: u_t = (U0*t^m*cos(pi/2), -U0*t^m*sin(pi/2))
    // Expected: idx=0 gives -U0*t^m*cos(pi/2) = -1.0*std::cos(1.5707963267948966) = 0.0
    //           idx=1 gives U0*t^m*sin(pi /2) = 1.0*std::sin(1.5707963267948966) = 1.0
    float_type u_x = svt(0, t, coord);
    float_type u_y = svt(1, t, coord);
    EXPECT_NEAR(u_x, 0.0, 1e-10);
    EXPECT_NEAR(u_y, 1.0, 1e-10);


}
// Test 2D pure rotation (no translation)
TEST(svt_test, pure_rotation_2d)
{
    // clang-format off
    std::string configStr = "                   \
        svt                                     \
        {                                       \
            p = 1.0;                            \
            m = -1.0;                           \
            Omega0 = 2.0;                       \
            U0 = 0.0;                           \
            alpha0 = 0.0;                       \
            x0 = 1.0;                           \
        }                                       \
    ";
    // clang-format on

    dictionary::Dictionary dict("svt", configStr);
    
    ib::SVT<2> svt;
    svt.init(&dict);

    float_type t = 2.0;
    types::vector_type<float_type, 2> coord(std::array<float_type, 2>{3.0, 1.5});

    // For pure rotation: u_r = (-Omega0*y*t^p, Omega0*(x-x0)*t^p)
    // With Omega0=2.0, t=2.0, p=1.0, x0=1.0
    // Expected: idx=0 gives -(-Omega0*y*t^p) = 2.0*1.5*2.0 = 6.0
    //           idx=1 gives -(Omega0*(x-x0)*t^p) = -2.0*(3.0-1.0)*2.0 = -8.0
    float_type u_x = svt(0, t, coord);
    float_type u_y = svt(1, t, coord);

    EXPECT_NEAR(u_x, 6.0, 1e-10);
    EXPECT_NEAR(u_y, -8.0, 1e-10);
}

// Test 2D combined translation and rotation
TEST(svt_test, combined_motion_2d)
{
    // clang-format off
    std::string configStr = "                   \
        svt                                     \
        {                                       \
            p = 1.0;                            \
            m = 1.0;                            \
            Omega0 = 2.0;                       \
            alpha0 = 1.5707963267948966;        \
            x0 = 0.0;                           \
        }                                       \
    ";
    // clang-format on

    dictionary::Dictionary dict("svt", configStr);
    
    ib::SVT<2> svt;
    svt.init(&dict);
    // std::cout<<svt.p<<" "<<svt.m<<" "<<svt.omega0<<" "<<svt.U0<<" "<<svt.alpha0<<" "<<svt.x0<<std::endl;
    float_type t = 1.0;
    types::vector_type<float_type, 2> coord(std::array<float_type, 2>{2.0, 3.0});

    // alpha0 = pi/2
    // alpha = alpha0 + 1/(p+1)*Omega0*t^(p+1) = pi/2 + 1/2*2.0*1.0^2 = pi/2 + 1.0
    // u_t = (U0*t^m*cos(alpha), -U0*t^m*sin(alpha))
    //     = (1.0*1.0*0, -1.0*1.0*-1.0) = (0, 1.0)
    // u_r = (-Omega0*y*t^p, Omega0*(x-x0)*t^p)
    //     = (-2.0*3.0*1.0, 2.0*2.0*1.0) = (-6.0, 4.0)
    // Total (negated): idx=0 gives -(0 - 6.0) = 6.0
    //                  idx=1 gives -(1.0 + 4.0) = -5.0

    float_type expected_alpha = 1.5707963267948966 + (1.0 / 2.0) * 2.0 * std::pow(1.0, 2.0);
    float_type u_t_x = 1.0 * 1.0 * std::cos(expected_alpha);
    float_type u_t_y = -1.0 * 1.0 * std::sin(expected_alpha);
    float_type u_r_x = -2.0 * 3.0 * std::pow(1.0, 1.0);
    float_type u_r_y = 2.0 * (2.0 - 0.0) * std::pow(1.0, 1.0);

    float_type u_x = svt(0, t, coord);
    float_type u_y = svt(1, t, coord);

    EXPECT_NEAR(u_x,-(u_t_x + u_r_x), 1e-6);
    EXPECT_NEAR(u_y, -(u_t_y + u_r_y), 1e-6);
}

// Test 2D with time-varying angle
TEST(svt_test, time_varying_angle_2d)
{
    // clang-format off
    std::string configStr = "                   \
        svt                                     \
        {                                       \
            p = 2.0;                            \
            m = 1.0;                            \
            Omega0 = 0.5;                       \
            U0 = 1.0;                           \
            alpha0 = 0.0;                       \
            x0 = 0.0;                           \
        }                                       \
    ";
    // clang-format on

    dictionary::Dictionary dict("svt", configStr);
    
    ib::SVT<2> svt;
    svt.init(&dict);

    float_type t = 2.0;
    types::vector_type<float_type, 2> coord(std::array<float_type, 2>{1.0, 1.0});

    // alpha = alpha0 + 1/(p+1) * omega0 * t^(p+1)
    //       = 0.0 + 1/3 * 0.5 * 2^3 = 1/3 * 0.5 * 8 = 4/3
    float_type expected_alpha = 0.0 + (1.0 / 3.0) * 0.5 * std::pow(2.0, 3.0);
    
    // u_t = (U0*t^m*cos(alpha), -U0*t^m*sin(alpha))
    //     = (1.0*2.0*cos(4/3), -1.0*2.0*sin(4/3))
    // u_r = (-Omega0*y*t^p, Omega0*x*t^p)
    //     = (-0.5*1.0*4.0, 0.5*1.0*4.0) = (-2.0, 2.0)
    float_type u_t_x = 1.0 * 2.0 * std::cos(expected_alpha);
    float_type u_t_y = -1.0 * 2.0 * std::sin(expected_alpha);
    float_type u_r_x = -0.5 * 1.0 * 4.0;
    float_type u_r_y = 0.5 * 1.0 * 4.0;

    float_type u_x = svt(0, t, coord);
    float_type u_y = svt(1, t, coord);

    EXPECT_NEAR(u_x, -(u_t_x + u_r_x), 1e-10);
    EXPECT_NEAR(u_y, -(u_t_y + u_r_y), 1e-10);
}

// Test 3D pure translation (no rotation)
TEST(svt_test, pure_translation_3d)
{
    // clang-format off
    std::string configStr = "                   \
        svt                                     \
        {                                       \
            p = -1.0;                           \
            m = 2.0;                            \
            Omega0 = 0.0;                       \
            U0 = 3.0;                           \
            alpha0 = 0.0;                       \
            x0 = 0.0;                           \
        }                                       \
    ";
    // clang-format on

    dictionary::Dictionary dict("svt", configStr);
    
    ib::SVT<3> svt;
    svt.init(&dict);

    float_type t = 2.0;
    types::vector_type<float_type, 3> coord(std::array<float_type, 3>{1.0, 2.0, 3.0});

    // For 3D pure translation with alpha0=0:
    // idx=0 (normal): -(-U0*t^m*sin(0)) = 0
    // idx=1: 0 (no y velocity)
    // idx=2 (tangent): -(U0*t^m*cos(0)) = -3.0*4.0*1.0 = -12.0
    float_type u_x = svt(0, t, coord);
    float_type u_y = svt(1, t, coord);
    float_type u_z = svt(2, t, coord);

    EXPECT_NEAR(u_x, 0.0, 1e-10);
    EXPECT_NEAR(u_y, 0.0, 1e-10);
    EXPECT_NEAR(u_z, -12.0, 1e-10);
}

// Test 3D pure translation at AoA (no rotation)
TEST(svt_test, pure_translation_aoa_3d)
{
    // clang-format off
    std::string configStr = "                   \
        svt                                     \
        {                                       \
            p = -1.0;                           \
            m = 0.0;                            \
            Omega0 = 0.0;                       \
            U0 = 1.0;                           \
            alpha0 = 0.7853981634;              \
            x0 = 0.0;                           \
        }                                       \
    ";
    // clang-format on

    dictionary::Dictionary dict("svt", configStr);
    ib::SVT<3> svt;
    svt.init(&dict);
    float_type t = 1.0;
    types::vector_type<float_type, 3> coord(std::array<float_type, 3>{0.5, 0.3, 0.7});
    // For pure translation with alpha0=pi/4: u_t = (-U0*t^m*sin(alpha), 0, U0*t^m*cos(alpha))
    // Expected: idx=0 gives -(-U0*t^m*sin(pi/4)) = 0.7071067812
    //           idx=1 gives 0
    //           idx=2 gives -(U0*t^m*cos(pi/4)) = -0.7071067812

    float_type u_x = svt(0, t, coord);
    float_type u_y = svt(1, t, coord);
    float_type u_z = svt(2, t, coord);
    EXPECT_NEAR(u_x, 0.7071067812, 1e-10);
    EXPECT_NEAR(u_y, 0.0, 1e-10);
    EXPECT_NEAR(u_z, -0.7071067812, 1e-10);
}

// Test 3D pure translation in x (no rotation) (normal to disk)
TEST(svt_test, pure_translation_x_3d)
{
    // clang-format off
    std::string configStr = "                   \
        svt                                     \
        {                                       \
            p = -1.0;                           \
            m = 0.0;                            \
            Omega0 = 0.0;                       \
            U0 = 1.0;                           \
            alpha0 = 1.5707963267948966;        \
            x0 = 0.0;                           \
        }                                       \
    ";
    // clang-format on

    dictionary::Dictionary dict("svt", configStr);
    ib::SVT<3> svt;
    svt.init(&dict);
    float_type t = 1.0;
    types::vector_type<float_type, 3> coord(std::array<float_type, 3>{0.5, 0.3, 0.7});
    // For pure translation with alpha0=pi/2: u_t = (-U0*t^m*sin(pi/2), 0, U0*t^m*cos(pi/2))
    // Expected: idx=0 gives -(-U0*t^m*sin(pi/2)) = 1.0
    //           idx=1 gives 0
    //           idx=2 gives -(U0*t^m*cos(pi/2)) = 0.0
    float_type u_x = svt(0, t, coord);
    float_type u_y = svt(1, t, coord);
    float_type u_z = svt(2, t, coord);
    EXPECT_NEAR(u_x, 1.0, 1e-10);
    EXPECT_NEAR(u_y, 0.0, 1e-10);
    EXPECT_NEAR(u_z, 0.0, 1e-10);
}

// Test 3D pure rotation (no translation)
TEST(svt_test, pure_rotation_3d)
{
    // clang-format off
    std::string configStr = "                   \
        svt                                     \
        {                                       \
            p = 1.0;                            \
            m = -1.0;                           \
            Omega0 = 1.5;                       \
            U0 = 0.0;                           \
            alpha0 = 0.0;                       \
            x0 = 2.0;                           \
        }                                       \
    ";
    // clang-format on

    dictionary::Dictionary dict("svt", configStr);
    
    ib::SVT<3> svt;
    svt.init(&dict);

    float_type t = 3.0;
    types::vector_type<float_type, 3> coord(std::array<float_type, 3>{4.0, 0.0, 5.0});

    // For 3D pure rotation:
    // idx=0: -(Omega0*(z-x0)*t^p) = -1.5*(5.0-2.0)*3.0 = -13.5
    // idx=1: 0
    // idx=2: -(-Omega0*x*t^p) = 1.5*4.0*3.0 = 18.0
    float_type u_x = svt(0, t, coord);
    float_type u_y = svt(1, t, coord);
    float_type u_z = svt(2, t, coord);

    EXPECT_NEAR(u_x, -13.5, 1e-10);
    EXPECT_NEAR(u_y, 0.0, 1e-10);
    EXPECT_NEAR(u_z, 18.0, 1e-10);
}

// Test 3D combined translation and rotation
TEST(svt_test, combined_motion_3d)
{
    // clang-format off
    std::string configStr = "                   \
        svt                                     \
        {                                       \
            p = 1.0;                            \
            m = 1.0;                            \
            Omega0 = 2.0;                       \
            U0 = 1.0;                           \
            alpha0 = 0.7853981633974483;        \
            x0 = 1.0;                           \
        }                                       \
    ";
    // clang-format on

    dictionary::Dictionary dict("svt", configStr);
    
    ib::SVT<3> svt;
    svt.init(&dict);

    float_type t = 1.0;
    types::vector_type<float_type, 3> coord(std::array<float_type, 3>{3.0, 0.0, 2.0});

    // alpha0 = pi/4
    // alpha= alpha0 + 1/(p+1)*Omega0*t^(p+1) = pi/4 + 1/2*2.0*1.0^2 = pi/4 + 1.0
    // u_t = (-U0*t^m*sin(alpha), 0, U0*t^m*cos(alpha))
    //     = (-1.0*1.0*sin(pi/4 + 1.0), 0, 1.0*1.0*cos(pi/4 + 1.0))
    // u_r = (Omega0*(z-x0)*t^p, 0, -Omega0*x*t^p)
    //     = (2.0*(2.0-1.0)*1.0, 0, -2.0*3.0*1.0) = (2.0, 0, -6.0)
    // Total (negated): idx=0 gives -( -1.0*1.0*sin(pi/4 + 1.0) + 2.0) = 1.0*sin(pi/4 + 1.0) - 2.0
    //                  idx=1 gives 0
    //                  idx=2 gives -(1.0*1.0*cos(pi/4 + 1.0) - 6.0) = -1.0*cos(pi/ 4 + 1.0) + 6.0  
    float_type expected_alpha = 0.7853981633974483 + (1.0 / 2.0) * 2.0 * std::pow(1.0, 2.0);
    float_type u_t_x = -1.0 * 1.0 * std::sin(expected_alpha);
    float_type u_t_z = 1.0 * 1.0 * std::cos(expected_alpha);
    float_type u_r_x = 2.0 * (2.0 - 1.0) * std::pow(1.0, 1.0);
    float_type u_r_z = -2.0 * 3.0 * std::pow(1.0, 1.0);
    




    float_type u_x = svt(0, t, coord);
    float_type u_y = svt(1, t, coord);
    float_type u_z = svt(2, t, coord);

    EXPECT_NEAR(u_x, -(u_t_x + u_r_x), 1e-10);
    EXPECT_NEAR(u_y, 0.0, 1e-10);
    EXPECT_NEAR(u_z, -(u_t_z + u_r_z), 1e-10);
}

// Test default parameters
TEST(svt_test, default_parameters)
{
    // clang-format off
    std::string configStr = "                   \
        svt                                     \
        {                                       \
        }                                       \
    ";
    // clang-format on

    dictionary::Dictionary dict("svt", configStr);
    
    ib::SVT<2> svt;
    svt.init(&dict);

    // Verify default parameters
    EXPECT_EQ(svt.p, 2.0);
    EXPECT_EQ(svt.m, 1.0);
    EXPECT_EQ(svt.omega0, 0.0);
    EXPECT_EQ(svt.U0, 1.0);
    EXPECT_EQ(svt.alpha0, 0.0);
    EXPECT_EQ(svt.x0, 0.0);
}

// Test invalid index throws exception (2D)
TEST(svt_test, invalid_index_2d)
{
    // clang-format off
    std::string configStr = "                   \
        svt                                     \
        {                                       \
            p = 1.0;                            \
            m = 1.0;                            \
            Omega0 = 1.0;                       \
            U0 = 1.0;                           \
            alpha0 = 0.0;                       \
            x0 = 0.0;                           \
        }                                       \
    ";
    // clang-format on

    dictionary::Dictionary dict("svt", configStr);
    
    ib::SVT<2> svt;
    svt.init(&dict);

    float_type t = 1.0;
    types::vector_type<float_type, 2> coord(std::array<float_type, 2>{0.0, 0.0});

    EXPECT_THROW(svt(2, t, coord), std::runtime_error);
}

// Test invalid index throws exception (3D)
TEST(svt_test, invalid_index_3d)
{
    // clang-format off
    std::string configStr = "                   \
        svt                                     \
        {                                       \
            p = 1.0;                            \
            m = 1.0;                            \
            Omega0 = 1.0;                       \
            U0 = 1.0;                           \
            alpha0 = 0.0;                       \
            x0 = 0.0;                           \
        }                                       \
    ";
    // clang-format on

    dictionary::Dictionary dict("svt", configStr);
    
    ib::SVT<3> svt;
    svt.init(&dict);

    float_type t = 1.0;
    types::vector_type<float_type, 3> coord(std::array<float_type, 3>{0.0, 0.0, 0.0});

    EXPECT_THROW(svt(3, t, coord), std::runtime_error);
}

// Test edge case: zero time
TEST(svt_test, zero_time)
{
    // clang-format off
    std::string configStr = "                   \
        svt                                     \
        {                                       \
            p = 1.0;                            \
            m = 1.0;                            \
            Omega0 = 1.0;                       \
            U0 = 2.0;                           \
            alpha0 = 0.0;                       \
            x0 = 0.0;                           \
        }                                       \
    ";
    // clang-format on

    dictionary::Dictionary dict("svt", configStr);
    
    ib::SVT<2> svt;
    svt.init(&dict);

    float_type t = 0.0;
    types::vector_type<float_type, 2> coord(std::array<float_type, 2>{1.0, 1.0});

    // At t=0, both translation and rotation should be zero
    float_type u_x = svt(0, t, coord);
    float_type u_y = svt(1, t, coord);

    EXPECT_NEAR(u_x, 0.0, 1e-10);
    EXPECT_NEAR(u_y, 0.0, 1e-10);
}

// Test power laws with different exponents
TEST(svt_test, power_law_exponents)
{
    // clang-format off
    std::string configStr = "                   \
        svt                                     \
        {                                       \
            p = 0.5;                            \
            m = 2.5;                            \
            Omega0 = 1.0;                       \
            U0 = 1.0;                           \
            alpha0 = 0.0;                       \
            x0 = 0.0;                           \
        }                                       \
    ";
    // clang-format on

    dictionary::Dictionary dict("svt", configStr);
    
    ib::SVT<3> svt;
    svt.init(&dict);

    float_type t = 4.0;
    types::vector_type<float_type, 3> coord(std::array<float_type, 3>{2.0, 3.0, 3.0});
    // alpha = alpha0 + 1/(p+1)*Omega0*t^(p+1) = 0.0 + 1/1.5*1.0*4^(1.5) = 1/1.5*8 = 16/3
    // u_t = (-U0*t^m*sin(alpha), 0, U0*t^m*cos(alpha)) = (-1.0*4.0^2.5*sin(16/3), 0, 1.0*4.0^2.5*cos(16/3))
    //     = (-128.0*sin(16/3), 0, 128.0*cos(16/3))
    // u_r = (Omega0*(z-x0)*t^p, 0, -Omega0*x*t^p) = (1.0*(3.0-0.0)*4.0^0.5, 0, -1.0*2.0*4.0^0.5) = (6.0, 0, -4.0)
    // Total (negated): idx=0 gives -(-128.0*sin(16/3) + 6.0) = 128.0*sin(16/3) - 6.0
    //                  idx=1 gives 0
    //                  idx=2 gives -(128.0*cos(16/3) - 4.0) = -128.0*cos(16/3) + 4.0   

    float_type expected_alpha = 0.0 + (1.0 / 1.5) * 1.0 * std::pow(4.0, 1.5);
    float_type u_t_x = -1.0 * std::pow(4.0, 2.5) * std::sin(expected_alpha);
    float_type u_t_z = 1.0 * std::pow(4.0, 2.5) * std::cos(expected_alpha);
    float_type u_r_x = 1.0 * (3.0 - 0.0) * std::pow(4.0, 0.5);
    float_type u_r_z = -1.0 * 2.0 * std::pow(4.0, 0.5); 

    float_type u_x = svt(0, t, coord);
    float_type u_y = svt(1, t, coord);

    EXPECT_NEAR(u_x, -(u_t_x + u_r_x), 1e-10);
    EXPECT_NEAR(u_y, 0.0, 1e-10);
}


} // namespace iblgf
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS(); // now MPI is already initialized
}