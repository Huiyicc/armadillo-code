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


//! \addtogroup operator_minus
//! @{



//! unary -
template<typename T1>
arma_inline
typename
enable_if2< (is_arma_type<T1>::value && is_signed<typename T1::elem_type>::value), const eOp<T1, eop_neg> >::result
operator-
(const T1& X)
  {
  arma_debug_sigprint();
  
  return eOp<T1,eop_neg>(X);
  }



template<typename T1>
arma_inline
typename enable_if2< (is_arma_type<T1>::value && (is_signed<typename T1::elem_type>::value == false)), const eOp<T1, eop_scalar_times> >::result
operator-
(const T1& X)
  {
  arma_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  return eOp<T1, eop_scalar_times>(X, eT(-1));
  }



//! Base - scalar
template<typename T1>
arma_inline
typename
enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_scalar_minus_post> >::result
operator-
  (
  const T1&                    X,
  const typename T1::elem_type k
  )
  {
  arma_debug_sigprint();
  
  return eOp<T1, eop_scalar_minus_post>(X, k);
  }



//! scalar - Base
template<typename T1>
arma_inline
typename
enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_scalar_minus_pre> >::result
operator-
  (
  const typename T1::elem_type k,
  const T1&                    X
  )
  {
  arma_debug_sigprint();
  
  return eOp<T1, eop_scalar_minus_pre>(X, k);
  }



//! complex scalar - non-complex Base
template<typename T1>
arma_inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_cx<typename T1::elem_type>::no),
  const mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_minus_pre>
  >::result
operator-
  (
  const std::complex<typename T1::pod_type>& k,
  const T1&                                  X
  )
  {
  arma_debug_sigprint();
  
  return mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_minus_pre>('j', X, k);
  }



//! non-complex Base - complex scalar
template<typename T1>
arma_inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_cx<typename T1::elem_type>::no),
  const mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_minus_post>
  >::result
operator-
  (
  const T1&                                  X,
  const std::complex<typename T1::pod_type>& k
  )
  {
  arma_debug_sigprint();
  
  return mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_minus_post>('j', X, k);
  }



//! subtraction of Base objects with same element type
template<typename T1, typename T2>
arma_inline
typename
enable_if2
  <
  is_arma_type<T1>::value && is_arma_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value,
  const eGlue<T1, T2, eglue_minus>
  >::result
operator-
  (
  const T1& X,
  const T2& Y
  )
  {
  arma_debug_sigprint();
  
  return eGlue<T1, T2, eglue_minus>(X, Y);
  }



//! subtraction of Base objects with different element types
template<typename T1, typename T2>
inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_arma_type<T2>::value && (is_same_type<typename T1::elem_type, typename T2::elem_type>::no)),
  const mtGlue<typename promote_type<typename T1::elem_type, typename T2::elem_type>::result, T1, T2, glue_mixed_minus>
  >::result
operator-
  (
  const T1& X,
  const T2& Y
  )
  {
  arma_debug_sigprint();
  
  typedef typename T1::elem_type eT1;
  typedef typename T2::elem_type eT2;
  
  typedef typename promote_type<eT1,eT2>::result out_eT;
  
  promote_type<eT1,eT2>::check();
  
  return mtGlue<out_eT, T1, T2, glue_mixed_minus>( X, Y );
  }



//! unary "-" for sparse objects 
template<typename T1>
inline
typename
enable_if2
  <
  is_arma_sparse_type<T1>::value && is_signed<typename T1::elem_type>::value,
  SpOp<T1,spop_scalar_times>
  >::result
operator-
(const T1& X)
  {
  arma_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  return SpOp<T1,spop_scalar_times>(X, eT(-1));
  }



//! subtraction of two sparse objects
template<typename T1, typename T2>
inline
typename
enable_if2
  <
  (is_arma_sparse_type<T1>::value && is_arma_sparse_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value),
  const SpGlue<T1,T2,spglue_minus>
  >::result
operator-
  (
  const T1& X,
  const T2& Y
  )
  {
  arma_debug_sigprint();
  
  return SpGlue<T1,T2,spglue_minus>(X,Y);
  }



//! subtraction of one sparse and one dense object
template<typename T1, typename T2>
inline
typename
enable_if2
  <
  (is_arma_sparse_type<T1>::value && is_arma_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value),
  Mat<typename T1::elem_type>
  >::result
operator-
  (
  const T1& x,
  const T2& y
  )
  {
  arma_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const SpProxy<T1> pa(x);
  
  const quasi_unwrap<T2> UB(y);
  const Mat<eT>& B     = UB.M;
  
  Mat<eT> result = -B;
  
  arma_conform_assert_same_size( pa.get_n_rows(), pa.get_n_cols(), result.n_rows, result.n_cols, "subtraction" );
  
  typename SpProxy<T1>::const_iterator_type it     = pa.begin();
  typename SpProxy<T1>::const_iterator_type it_end = pa.end();
  
  for(; it != it_end; ++it)
    {
    const uword r = it.row();
    const uword c = it.col();
    
    result.at(r, c) = (*it) - B.at(r,c);
    }
  
  return result;
  }



//! subtraction of one dense and one sparse object
template<typename T1, typename T2>
inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_arma_sparse_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value),
  Mat<typename T1::elem_type>
  >::result
operator-
  (
  const T1& x,
  const T2& y
  )
  {
  arma_debug_sigprint();
  
  Mat<typename T1::elem_type> result(x);
  
  const SpProxy<T2> pb(y);
  
  arma_conform_assert_same_size( result.n_rows, result.n_cols, pb.get_n_rows(), pb.get_n_cols(), "subtraction" );
  
  typename SpProxy<T2>::const_iterator_type it     = pb.begin();
  typename SpProxy<T2>::const_iterator_type it_end = pb.end();

  while(it != it_end)
    {
    result.at(it.row(), it.col()) -= (*it);
    ++it;
    }
  
  return result;
  }



//! subtraction of two sparse objects with different element types
template<typename T1, typename T2>
inline
typename
enable_if2
  <
  (is_arma_sparse_type<T1>::value && is_arma_sparse_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::no),
  const mtSpGlue< typename promote_type<typename T1::elem_type, typename T2::elem_type>::result, T1, T2, spglue_minus_mixed >
  >::result
operator-
  (
  const T1& X,
  const T2& Y
  )
  {
  arma_debug_sigprint();
  
  typedef typename T1::elem_type eT1;
  typedef typename T2::elem_type eT2;
  
  typedef typename promote_type<eT1,eT2>::result out_eT;
  
  promote_type<eT1,eT2>::check();
  
  return mtSpGlue<out_eT, T1, T2, spglue_minus_mixed>( X, Y );
  }



//! subtraction of sparse and non-sparse objects with different element types
template<typename T1, typename T2>
inline
typename
enable_if2
  <
  (is_arma_sparse_type<T1>::value && is_arma_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::no),
  Mat< typename promote_type<typename T1::elem_type, typename T2::elem_type>::result >
  >::result
operator-
  (
  const T1& x,
  const T2& y
  )
  {
  arma_debug_sigprint();
  
  Mat< typename promote_type<typename T1::elem_type, typename T2::elem_type>::result > out;
  
  spglue_minus_mixed::sparse_minus_dense(out, x, y);
  
  return out;
  }



//! subtraction of sparse and non-sparse objects with different element types
template<typename T1, typename T2>
inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_arma_sparse_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::no),
  Mat< typename promote_type<typename T1::elem_type, typename T2::elem_type>::result >
  >::result
operator-
  (
  const T1& x,
  const T2& y
  )
  {
  arma_debug_sigprint();
  
  Mat< typename promote_type<typename T1::elem_type, typename T2::elem_type>::result > out;
  
  spglue_minus_mixed::dense_minus_sparse(out, x, y);
  
  return out;
  }



//! sparse - scalar
template<typename T1>
arma_inline
typename
enable_if2< is_arma_sparse_type<T1>::value, const SpToDOp<T1, op_sp_minus_post> >::result
operator-
  (
  const T1&                    X,
  const typename T1::elem_type k
  )
  {
  arma_debug_sigprint();
  
  return SpToDOp<T1, op_sp_minus_post>(X, k);
  }



//! scalar - sparse
template<typename T1>
arma_inline
typename
enable_if2< is_arma_sparse_type<T1>::value, const SpToDOp<T1, op_sp_minus_pre> >::result
operator-
  (
  const typename T1::elem_type k,
  const T1&                    X
  )
  {
  arma_debug_sigprint();
  
  return SpToDOp<T1, op_sp_minus_pre>(X, k);
  }



// TODO: this is an uncommon use case; remove?
//! multiple applications of add/subtract scalars can be condensed
template<typename T1, typename op_type>
inline
typename
enable_if2
  <
  (is_arma_sparse_type<T1>::value &&
      (is_same_type<op_type, op_sp_plus>::value ||
       is_same_type<op_type, op_sp_minus_post>::value)),
  const SpToDOp<T1, op_sp_minus_post>
  >::result
operator-
  (
  const SpToDOp<T1, op_type>&  x,
  const typename T1::elem_type k
  )
  {
  arma_debug_sigprint();

  const typename T1::elem_type aux = (is_same_type<op_type, op_sp_plus>::value) ? -x.aux : x.aux;

  return SpToDOp<T1, op_sp_minus_post>(x.m, aux + k);
  }



// TODO: this is an uncommon use case; remove?
//! multiple applications of add/subtract scalars can be condensed
template<typename T1, typename op_type>
inline
typename
enable_if2
  <
  (is_arma_sparse_type<T1>::value &&
      (is_same_type<op_type, op_sp_plus>::value ||
       is_same_type<op_type, op_sp_minus_post>::value)),
  const SpToDOp<T1, op_sp_minus_pre>
  >::result
operator-
  (
  const typename T1::elem_type k,
  const SpToDOp<T1, op_type>&  x
  )
  {
  arma_debug_sigprint();

  const typename T1::elem_type aux = (is_same_type<op_type, op_sp_plus>::value) ? -x.aux : x.aux;

  return SpToDOp<T1, op_sp_minus_pre>(x.m, k + aux);
  }



// TODO: this is an uncommon use case; remove?
//! multiple applications of add/subtract scalars can be condensed
template<typename T1, typename op_type>
inline
typename
enable_if2
  <
  (is_arma_sparse_type<T1>::value &&
       is_same_type<op_type, op_sp_minus_pre>::value),
  const SpToDOp<T1, op_sp_minus_pre>
  >::result
operator-
  (
  const SpToDOp<T1, op_type>&  x,
  const typename T1::elem_type k
  )
  {
  arma_debug_sigprint();

  return SpToDOp<T1, op_sp_minus_pre>(x.m, x.aux - k);
  }



// TODO: this is an uncommon use case; remove?
//! multiple applications of add/subtract scalars can be condensed
template<typename T1, typename op_type>
inline
typename
enable_if2
  <
  (is_arma_sparse_type<T1>::value &&
       is_same_type<op_type, op_sp_minus_pre>::value),
  const SpToDOp<T1, op_sp_plus>
  >::result
operator-
  (
  const typename T1::elem_type k,
  const SpToDOp<T1, op_type>&  x
  )
  {
  arma_debug_sigprint();

  return SpToDOp<T1, op_sp_plus>(x.m, k - x.aux);
  }



template<typename parent, unsigned int mode, typename T2>
arma_inline
Mat<typename parent::elem_type>
operator-
  (
  const subview_each1<parent,mode>&          X,
  const Base<typename parent::elem_type,T2>& Y
  )
  {
  arma_debug_sigprint();
  
  return subview_each1_aux::operator_minus(X, Y.get_ref());
  }



template<typename T1, typename parent, unsigned int mode>
arma_inline
Mat<typename parent::elem_type>
operator-
  (
  const Base<typename parent::elem_type,T1>& X,
  const subview_each1<parent,mode>&          Y
  )
  {
  arma_debug_sigprint();
  
  return subview_each1_aux::operator_minus(X.get_ref(), Y);
  }



template<typename parent, unsigned int mode, typename TB, typename T2>
arma_inline
Mat<typename parent::elem_type>
operator-
  (
  const subview_each2<parent,mode,TB>&       X,
  const Base<typename parent::elem_type,T2>& Y
  )
  {
  arma_debug_sigprint();
  
  return subview_each2_aux::operator_minus(X, Y.get_ref());
  }



template<typename T1, typename parent, unsigned int mode, typename TB>
arma_inline
Mat<typename parent::elem_type>
operator-
  (
  const Base<typename parent::elem_type,T1>& X,
  const subview_each2<parent,mode,TB>&       Y
  )
  {
  arma_debug_sigprint();
  
  return subview_each2_aux::operator_minus(X.get_ref(), Y);
  }



//! @}
