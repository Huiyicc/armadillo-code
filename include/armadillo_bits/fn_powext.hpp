// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
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


//! \addtogroup fn_powext
//! @{



template<typename T1, typename T2>
arma_warn_unused
inline
typename
enable_if2
  <
  is_arma_type<T1>::value,
  const Glue<T1, T2, glue_powext>
  >::result
pow
  (
  const T1&                               X,
  const Base<typename T1::elem_type, T2>& Y
  )
  {
  arma_debug_sigprint();
  
  return Glue<T1, T2, glue_powext>(X, Y.get_ref());
  }



template<typename T1, typename T2>
arma_warn_unused
inline
typename
enable_if2
  <
  ( is_arma_type<T1>::value && is_cx<typename T1::elem_type>::yes ),
  const mtGlue<typename T1::elem_type, T1, T2, glue_powext_cx>
  >::result
pow
  (
  const T1&                              X,
  const Base<typename T1::pod_type, T2>& Y
  )
  {
  arma_debug_sigprint();
  
  return mtGlue<typename T1::elem_type, T1, T2, glue_powext_cx>(X, Y.get_ref());
  }



template<typename T1, typename T2>
arma_warn_unused
inline
const GlueCube<T1, T2, glue_powext>
pow
  (
  const BaseCube<typename T1::elem_type, T1>& X,
  const BaseCube<typename T1::elem_type, T2>& Y
  )
  {
  arma_debug_sigprint();
  
  return GlueCube<T1, T2, glue_powext>(X.get_ref(), Y.get_ref());
  }



template<typename T1, typename T2>
arma_warn_unused
inline
Cube<typename T1::elem_type>
pow
  (
  const BaseCube<typename T1::elem_type, T1>& X,
  const Base    <typename T1::elem_type, T2>& Y
  )
  {
  arma_debug_sigprint();
  
  // rudimentary handling of broadcasting operations
  // mainly for compat with previous ill-designed direct handling of .each_slice()
  
  typedef typename T1::elem_type eT;
  
  Cube<eT> A = X.get_ref();
  
  const unwrap<T2>   UY(Y.get_ref());
  const Mat<eT>& B = UY.M;
  
  arma_conform_assert_same_size(A.n_rows, A.n_cols, B.n_rows, B.n_cols, "element-wise pow()");
  
  A.each_slice( [](Mat<eT>& S){ S = pow(S, B); } );
  
  return A;
  }



template<typename T1, typename T2>
arma_warn_unused
inline
const mtGlueCube<typename T1::elem_type, T1, T2, glue_powext_cx>
pow
  (
  const BaseCube< std::complex<typename T1::pod_type>, T1>& X,
  const BaseCube<              typename T1::pod_type , T2>& Y
  )
  {
  arma_debug_sigprint();
  
  return mtGlueCube<typename T1::elem_type, T1, T2, glue_powext_cx>(X.get_ref(), Y.get_ref());
  }



//! @}
