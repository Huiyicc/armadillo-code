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



//! \addtogroup op_powmat
//! @{


template<typename T1>
inline
void
op_powmat::apply(Mat<typename T1::elem_type>& out, const Op<T1, op_powmat>& expr)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword y     =  expr.aux_uword_a;
  const bool  y_neg = (expr.aux_uword_b == uword(1));
  
  if(y_neg)
    {
    if(y == uword(1))
      {
      const bool inv_status = inv(out, expr.m);
      
      if(inv_status == false)
        {
        out.soft_reset();
        arma_stop_runtime_error("powmat(): matrix inverse failed");
        return;
        }
      }
    else
      {
      Mat<eT> X_inv;
      
      const bool inv_status = inv(X_inv, expr.m);
      
      if(inv_status == false)
        {
        out.soft_reset();
        arma_stop_runtime_error("powmat(): matrix inverse failed");
        return;
        }
      
      op_powmat::apply(out, X_inv, y);
      }
    }
  else
    {
    const quasi_unwrap<T1> U(expr.m);
    
    arma_debug_check( (U.M.is_square() == false), "powmat(): given matrix must be square sized" );
    
    op_powmat::apply(out, U.M, y);
    }
  }



template<typename eT>
inline
void
op_powmat::apply(Mat<eT>& out, const Mat<eT>& X, const uword y)
  {
  arma_extra_debug_sigprint();
  
  const uword N = X.n_rows;
  
  if(y == uword(0))  { out.eye(N,N); return; }
  if(y == uword(1))  { out = X;      return; }
  
  if(X.is_diagmat())
    {
    // use temporary array in case we have aliasing
    
    podarray<eT> tmp(N);
    
    for(uword i=0; i<N; ++i)  { tmp[i] = eop_aux::pow(X.at(i,i), y); }  // TODO: y may need to converted to int or sword
    
    out.zeros(N,N);
    
    for(uword i=0; i<N; ++i)  { out.at(i,i) = tmp[i]; }
    }
  else
    {
         if(y == uword(2))  {                          out = X*X;       }
    else if(y == uword(3))  { const Mat<eT> tmp = X*X; out = X*tmp;     }
    else if(y == uword(4))  { const Mat<eT> tmp = X*X; out =   tmp*tmp; }
    else if(y == uword(5))  { const Mat<eT> tmp = X*X; out = X*tmp*tmp; }
    else
      {
      Mat<eT> tmp = X;
      
      out = X;
      
      uword z = y-1;
      
      while(z > 0)
        {
        if(z & 1)  { out = tmp * out; }
        
        z /= uword(2);
        
        if(z > 0)  { tmp = tmp * tmp; }
        }
      }
    }
  }


//! @}
