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


//! \addtogroup fn_reshape
//! @{



template<typename T1>
arma_warn_unused
inline
typename enable_if2< is_arma_type<T1>::value, const Op<T1, op_reshape> >::result
reshape(const T1& X, const uword new_n_rows, const uword new_n_cols)
  {
  arma_extra_debug_sigprint();
  
  return Op<T1, op_reshape>(X, new_n_rows, new_n_cols);
  }



template<typename T1>
arma_warn_unused
inline
typename enable_if2< is_arma_type<T1>::value, const Op<T1, op_reshape> >::result
reshape(const T1& X, const SizeMat& s)
  {
  arma_extra_debug_sigprint();
  
  return Op<T1, op_reshape>(X, s.n_rows, s.n_cols);
  }



//! NOTE: don't use this form: it will be removed
template<typename T1>
[[deprecated]]
inline
const Op<T1, op_reshape_old>
reshape(const Base<typename T1::elem_type,T1>& X, const uword new_n_rows, const uword new_n_cols, const uword dim)  //!< NOTE: don't use this form: it will be removed
  {
  arma_extra_debug_sigprint();
  
  // arma_debug_warn_level(1, "this form of reshape() is deprecated and will be removed");
  
  arma_debug_check( (dim > 1), "reshape(): parameter 'dim' must be 0 or 1" );
  
  return Op<T1, op_reshape_old>(X.get_ref(), new_n_rows, new_n_cols, dim, 'j');
  }



template<typename T1>
arma_warn_unused
inline
const OpCube<T1, op_reshape>
reshape(const BaseCube<typename T1::elem_type,T1>& X, const uword new_n_rows, const uword new_n_cols, const uword new_n_slices)
  {
  arma_extra_debug_sigprint();
  
  return OpCube<T1, op_reshape>(X.get_ref(), new_n_rows, new_n_cols, new_n_slices);
  }



template<typename T1>
arma_warn_unused
inline
const OpCube<T1, op_reshape>
reshape(const BaseCube<typename T1::elem_type,T1>& X, const SizeCube& s)
  {
  arma_extra_debug_sigprint();
  
  return OpCube<T1, op_reshape>(X.get_ref(), s.n_rows, s.n_cols, s.n_slices);
  }



template<typename T1>
arma_warn_unused
inline
const SpOp<T1, spop_reshape>
reshape(const SpBase<typename T1::elem_type, T1>& X, const uword new_n_rows, const uword new_n_cols)
  {
  arma_extra_debug_sigprint();
  
  return SpOp<T1, spop_reshape>(X.get_ref(), new_n_rows, new_n_cols);
  }



template<typename T1>
arma_warn_unused
inline
const SpOp<T1, spop_reshape>
reshape(const SpBase<typename T1::elem_type, T1>& X, const SizeMat& s)
  {
  arma_extra_debug_sigprint();
  
  return SpOp<T1, spop_reshape>(X.get_ref(), s.n_rows, s.n_cols);
  }



//! @}
