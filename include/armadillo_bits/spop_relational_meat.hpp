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


//! \addtogroup spop_relational
//! @{

// NOTE: relational operations between sparse matrices and scalars are generally not advised
// NOTE: due to the risk of producing sparse matrices full of non-zeros, gobbling up memory.
// NOTE: the implementations below are rudimentary, and only intended for completeness.
// NOTE: 
// NOTE: relational operations between sparse matrices and scalars should only be used
// NOTE: as an argument to the accu() function, which omits the generation of intermediate
// NOTE: sparse matrices.

template<typename T1>
inline
void
spop_rel_lt_pre::apply(SpMat<uword>& out, const mtSpOp<uword, T1, spop_rel_lt_pre>& X)
  {
  arma_debug_sigprint();
  
  // operation: scalar < spmat
  
  typedef typename T1::elem_type eT;
  
  const eT k = X.aux;
  
  const unwrap_spmat<T1> U(X.m);
  const SpMat<eT>& A =   U.M;
  
  if(U.is_alias(out))  { const SpMat<eT> tmp(U.M); out = (k < tmp); return; }
  
  if(arma_config::warn_level >= 2)
    {
    const uword A_n_zeros = A.n_elem - A.n_nonzero;
    
    const uword out_nnz_min = (k < eT(0)) ? A_n_zeros : 0;
    
    if( (out_nnz_min > (A.n_elem/2)) && (A.n_rows > 1) && (A.n_cols > 1) )
      {
      arma_warn(2, "relational comparison: resulting sparse matrix has more than 50% non-zeros");
      }
    }
  
  const uword n_rows = A.n_rows;
  const uword n_cols = A.n_cols;
  
  out.zeros(n_rows, n_cols);
  
  for(uword c=0; c < n_cols; ++c)
  for(uword r=0; r < n_rows; ++r)
    {
    if(k < A.at(r,c)) { out.at(r,c) = uword(1); }
    }
  }



template<typename T1>
inline
void
spop_rel_gt_pre::apply(SpMat<uword>& out, const mtSpOp<uword, T1, spop_rel_gt_pre>& X)
  {
  arma_debug_sigprint();
  
  // operation: scalar > spmat
  
  typedef typename T1::elem_type eT;
  
  const eT k = X.aux;
  
  const unwrap_spmat<T1> U(X.m);
  const SpMat<eT>& A =   U.M;
  
  if(U.is_alias(out))  { const SpMat<eT> tmp(U.M); out = (k > tmp); return; }
  
  if(arma_config::warn_level >= 2)
    {
    const uword A_n_zeros = A.n_elem - A.n_nonzero;
    
    const uword out_nnz_min = (k > eT(0)) ? A_n_zeros : 0;
    
    if( (out_nnz_min > (A.n_elem/2)) && (A.n_rows > 1) && (A.n_cols > 1) )
      {
      arma_warn(2, "relational comparison: resulting sparse matrix has more than 50% non-zeros");
      }
    }
  
  const uword n_rows = A.n_rows;
  const uword n_cols = A.n_cols;
  
  out.zeros(n_rows, n_cols);
  
  for(uword c=0; c < n_cols; ++c)
  for(uword r=0; r < n_rows; ++r)
    {
    if(k > A.at(r,c)) { out.at(r,c) = uword(1); }
    }
  }



template<typename T1>
inline
void
spop_rel_lteq_pre::apply(SpMat<uword>& out, const mtSpOp<uword, T1, spop_rel_lteq_pre>& X)
  {
  arma_debug_sigprint();
  
  // operation: scalar <= spmat
  
  typedef typename T1::elem_type eT;
  
  const eT k = X.aux;
  
  const unwrap_spmat<T1> U(X.m);
  const SpMat<eT>& A =   U.M;
  
  if(U.is_alias(out))  { const SpMat<eT> tmp(U.M); out = (k <= tmp); return; }
  
  if(arma_config::warn_level >= 2)
    {
    const uword A_n_zeros = A.n_elem - A.n_nonzero;
    
    const uword out_nnz_min = (k <= eT(0)) ? A_n_zeros : 0;
    
    if( (out_nnz_min > (A.n_elem/2)) && (A.n_rows > 1) && (A.n_cols > 1) )
      {
      arma_warn(2, "relational comparison: resulting sparse matrix has more than 50% non-zeros");
      }
    }
  
  const uword n_rows = A.n_rows;
  const uword n_cols = A.n_cols;
  
  out.zeros(n_rows, n_cols);
  
  for(uword c=0; c < n_cols; ++c)
  for(uword r=0; r < n_rows; ++r)
    {
    if(k <= A.at(r,c)) { out.at(r,c) = uword(1); }
    }
  }



template<typename T1>
inline
void
spop_rel_gteq_pre::apply(SpMat<uword>& out, const mtSpOp<uword, T1, spop_rel_gteq_pre>& X)
  {
  arma_debug_sigprint();
  
  // operation: scalar >= spmat
  
  typedef typename T1::elem_type eT;
  
  const eT k = X.aux;
  
  const unwrap_spmat<T1> U(X.m);
  const SpMat<eT>& A =   U.M;
  
  // TODO: optimisation for spmat >= positive_nonzero_value
  
  if(U.is_alias(out))  { const SpMat<eT> tmp(U.M); out = (k >= tmp); return; }
  
  if(arma_config::warn_level >= 2)
    {
    const uword A_n_zeros = A.n_elem - A.n_nonzero;
    
    const uword out_nnz_min = (k >= eT(0)) ? A_n_zeros : 0;
    
    if( (out_nnz_min > (A.n_elem/2)) && (A.n_rows > 1) && (A.n_cols > 1) )
      {
      arma_warn(2, "relational comparison: resulting sparse matrix has more than 50% non-zeros");
      }
    }
  
  const uword n_rows = A.n_rows;
  const uword n_cols = A.n_cols;
  
  out.zeros(n_rows, n_cols);
  
  for(uword c=0; c < n_cols; ++c)
  for(uword r=0; r < n_rows; ++r)
    {
    if(k >= A.at(r,c)) { out.at(r,c) = uword(1); }
    }
  }



template<typename T1>
inline
void
spop_rel_lt_post::apply(SpMat<uword>& out, const mtSpOp<uword, T1, spop_rel_lt_post>& X)
  {
  arma_debug_sigprint();
  
  // operation: spmat < scalar
  
  typedef typename T1::elem_type eT;
  
  const eT k = X.aux;
  
  const unwrap_spmat<T1> U(X.m);
  const SpMat<eT>& A =   U.M;
  
  // TODO: optimisation for spmat < 0
  
  if(U.is_alias(out))  { const SpMat<eT> tmp(U.M); out = (tmp < k); return; }
  
  if(arma_config::warn_level >= 2)
    {
    const uword A_n_zeros = A.n_elem - A.n_nonzero;
    
    const uword out_nnz_min = (eT(0) < k) ? A_n_zeros : 0;
    
    if( (out_nnz_min > (A.n_elem/2)) && (A.n_rows > 1) && (A.n_cols > 1) )
      {
      arma_warn(2, "relational comparison: resulting sparse matrix has more than 50% non-zeros");
      }
    }
  
  const uword n_rows = A.n_rows;
  const uword n_cols = A.n_cols;
  
  out.zeros(n_rows, n_cols);
  
  for(uword c=0; c < n_cols; ++c)
  for(uword r=0; r < n_rows; ++r)
    {
    if(A.at(r,c) < k) { out.at(r,c) = uword(1); }
    }
  }



template<typename T1>
inline
void
spop_rel_gt_post::apply(SpMat<uword>& out, const mtSpOp<uword, T1, spop_rel_gt_post>& X)
  {
  arma_debug_sigprint();
  
  // operation: spmat > scalar
  
  typedef typename T1::elem_type eT;
  
  const eT k = X.aux;
  
  const unwrap_spmat<T1> U(X.m);
  const SpMat<eT>& A =   U.M;
  
  // TODO: optimisation for spmat > 0  -> this is equivalent to spones(A)
  
  // TODO: optimisation for spmat > positive_nonzero_value
  
  if(U.is_alias(out))  { const SpMat<eT> tmp(U.M); out = (tmp > k); return; }
  
  if(arma_config::warn_level >= 2)
    {
    const uword A_n_zeros = A.n_elem - A.n_nonzero;
    
    const uword out_nnz_min = (eT(0) > k) ? A_n_zeros : 0;
    
    if( (out_nnz_min > (A.n_elem/2)) && (A.n_rows > 1) && (A.n_cols > 1) )
      {
      arma_warn(2, "relational comparison: resulting sparse matrix has more than 50% non-zeros");
      }
    }
  
  const uword n_rows = A.n_rows;
  const uword n_cols = A.n_cols;
  
  out.zeros(n_rows, n_cols);
  
  for(uword c=0; c < n_cols; ++c)
  for(uword r=0; r < n_rows; ++r)
    {
    if(A.at(r,c) > k) { out.at(r,c) = uword(1); }
    }
  }



template<typename T1>
inline
void
spop_rel_lteq_post::apply(SpMat<uword>& out, const mtSpOp<uword, T1, spop_rel_lteq_post>& X)
  {
  arma_debug_sigprint();
  
  // operation: spmat <= scalar
  
  typedef typename T1::elem_type eT;
  
  const eT k = X.aux;
  
  const unwrap_spmat<T1> U(X.m);
  const SpMat<eT>& A =   U.M;
  
  if(U.is_alias(out))  { const SpMat<eT> tmp(U.M); out = (tmp <= k); return; }
  
  if(arma_config::warn_level >= 2)
    {
    const uword A_n_zeros = A.n_elem - A.n_nonzero;
    
    const uword out_nnz_min = (eT(0) <= k) ? A_n_zeros : 0;
    
    if( (out_nnz_min > (A.n_elem/2)) && (A.n_rows > 1) && (A.n_cols > 1) )
      {
      arma_warn(2, "relational comparison: resulting sparse matrix has more than 50% non-zeros");
      }
    }
  
  const uword n_rows = A.n_rows;
  const uword n_cols = A.n_cols;
  
  out.zeros(n_rows, n_cols);
  
  for(uword c=0; c < n_cols; ++c)
  for(uword r=0; r < n_rows; ++r)
    {
    if(A.at(r,c) <= k) { out.at(r,c) = uword(1); }
    }
  }



template<typename T1>
inline
void
spop_rel_gteq_post::apply(SpMat<uword>& out, const mtSpOp<uword, T1, spop_rel_gteq_post>& X)
  {
  arma_debug_sigprint();
  
  // operation: spmat >= scalar
  
  typedef typename T1::elem_type eT;
  
  const eT k = X.aux;
  
  const unwrap_spmat<T1> U(X.m);
  const SpMat<eT>& A =   U.M;
  
  if(U.is_alias(out))  { const SpMat<eT> tmp(U.M); out = (tmp >= k); return; }
  
  if(arma_config::warn_level >= 2)
    {
    const uword A_n_zeros = A.n_elem - A.n_nonzero;
    
    const uword out_nnz_min = (eT(0) >= k) ? A_n_zeros : 0;
    
    if( (out_nnz_min > (A.n_elem/2)) && (A.n_rows > 1) && (A.n_cols > 1) )
      {
      arma_warn(2, "relational comparison: resulting sparse matrix has more than 50% non-zeros");
      }
    }
  
  const uword n_rows = A.n_rows;
  const uword n_cols = A.n_cols;
  
  out.zeros(n_rows, n_cols);
  
  for(uword c=0; c < n_cols; ++c)
  for(uword r=0; r < n_rows; ++r)
    {
    if(A.at(r,c) >= k) { out.at(r,c) = uword(1); }
    }
  }



template<typename T1>
inline
void
spop_rel_eq::apply(SpMat<uword>& out, const mtSpOp<uword, T1, spop_rel_eq>& X)
  {
  arma_debug_sigprint();
  
  // operation: spmat == scalar
  
  typedef typename T1::elem_type eT;
  
  const eT k = X.aux;
  
  const unwrap_spmat<T1> U(X.m);
  const SpMat<eT>& A =   U.M;
  
  if(U.is_alias(out))  { const SpMat<eT> tmp(U.M); out = (tmp == k); return; }
  
  if(arma_config::warn_level >= 2)
    {
    const uword A_n_zeros = A.n_elem - A.n_nonzero;
    
    const uword out_nnz_min = (eT(0) == k) ? A_n_zeros : 0;
    
    if( (out_nnz_min > (A.n_elem/2)) && (A.n_rows > 1) && (A.n_cols > 1) )
      {
      arma_warn(2, "relational comparison: resulting sparse matrix has more than 50% non-zeros");
      }
    }
  
  const uword n_rows = A.n_rows;
  const uword n_cols = A.n_cols;
  
  out.zeros(n_rows, n_cols);
  
  for(uword c=0; c < n_cols; ++c)
  for(uword r=0; r < n_rows; ++r)
    {
    if(A.at(r,c) == k) { out.at(r,c) = uword(1); }
    }
  }



template<typename T1>
inline
void
spop_rel_noteq::apply(SpMat<uword>& out, const mtSpOp<uword, T1, spop_rel_noteq>& X)
  {
  arma_debug_sigprint();
  
  // operation: spmat != scalar
  
  typedef typename T1::elem_type eT;
  
  const eT k = X.aux;
  
  const unwrap_spmat<T1> U(X.m);
  const SpMat<eT>& A =   U.M;
  
  if(U.is_alias(out))  { const SpMat<eT> tmp(U.M); out = (tmp != k); return; }
  
  if(arma_config::warn_level >= 2)
    {
    const uword A_n_zeros = A.n_elem - A.n_nonzero;
    
    const uword out_nnz_min = (eT(0) != k) ? A_n_zeros : 0;
    
    if( (out_nnz_min > (A.n_elem/2)) && (A.n_rows > 1) && (A.n_cols > 1) )
      {
      arma_warn(2, "relational comparison: resulting sparse matrix has more than 50% non-zeros");
      }
    }
  
  const uword n_rows = A.n_rows;
  const uword n_cols = A.n_cols;
  
  out.zeros(n_rows, n_cols);
  
  for(uword c=0; c < n_cols; ++c)
  for(uword r=0; r < n_rows; ++r)
    {
    if(A.at(r,c) != k) { out.at(r,c) = uword(1); }
    }
  }



//! @}
