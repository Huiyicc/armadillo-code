// Copyright 2015 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2015 National ICT Australia (NICTA)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include <algorithm>

#include <armadillo>
#include "catch.hpp"

using namespace arma;

template <typename MatType>
void qr_check(const MatType& Q, const MatType& R, const MatType &A, double atol, double rtol,
              uvec* P_ptr = nullptr) {
    MatType I;
    I.eye(A.n_rows, A.n_rows);

    // Q should have same number of rows as A
    REQUIRE(Q.n_rows == A.n_rows);
    // Q should be square
    REQUIRE(Q.n_cols == Q.n_rows);
    // Q should be orthogonal
    REQUIRE(approx_equal(Q.t() * Q, I, "both", atol, rtol));
    REQUIRE(approx_equal(Q * Q.t(), I, "both", atol, rtol));

    // R should be upper-triangular
    for (uword col_idx = 0; col_idx < R.n_cols; ++col_idx) {
        for (uword row_idx = col_idx + 1; row_idx < R.n_rows; ++row_idx) {
            REQUIRE(std::abs(R.at(row_idx, col_idx)) < atol);
        }
    }

    if (P_ptr) {
        // this is a pivoted QR
        REQUIRE(approx_equal(Q * R, A.cols(*P_ptr), "both", atol, rtol));
    } else {
        // this is a normal QR
        REQUIRE(approx_equal(Q * R, A, "both", atol, rtol));
    }
}

TEST_CASE("fn_qr_real_1")
  {
  mat A =
    "\
      1 -1  0;\
     -1  3  0;\
      0  1  0;\
    ";
  double atol = 1e-10;
  double rtol = 1e-8;

  mat Q, R;
  qr(Q, R, A);
  qr_check(Q, R, A, atol, rtol);
  }


TEST_CASE("fn_qr_pivot_real_1")
  {
  mat A =
    "\
      1 -1  0;\
     -1  3  0;\
      0  1  0;\
    ";
  double atol = 1e-10;
  double rtol = 1e-8;

  mat Q, R;
  uvec P;
  qr(Q, R, P, A);
  qr_check(Q, R, A, atol, rtol, &P);
  }

TEST_CASE("fn_qr_pivot_complex_1")
  {
  cx_mat A =
    "\
      1+1j   -1  0;\
     -1    3-1j  0;\
      0       1  0;\
    ";
  double atol = 1e-10;
  double rtol = 1e-8;

  cx_mat Q, R;
  uvec P;
  qr(Q, R, P, A);
  qr_check(Q, R, A, atol, rtol, &P);
  }
