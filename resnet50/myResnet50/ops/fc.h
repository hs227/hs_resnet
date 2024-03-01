#pragma once
#ifndef FC_H
#define FC_H

#include"normal.h"



//// Y=W*X+B
//// X,Y,B都是列向量
//template<typename T>
//vector<vector<T>> fc2DCol(vector<vector<T>>& X,
//  vector<vector<T>>& W, vector<vector<T>>& B) 
//{
//  const unsigned int hx = X.size();
//  const unsigned int wx = X[0].size();
//  const unsigned int hw = W.size();
//  const unsigned int ww = W[0].size();
//  const unsigned int hb = B.size();
//  const unsigned int wb = B[0].size();
//
//  const unsigned int hy = hw;
//  const unsigned int wy = wx;
//
//  vector<vector<T>> Y;
//  Y = mat2DMul(W, X);
//  Y = mat2DAdd(Y, B);
//  return Y;
//}
//template<typename T>
//T* fc2DCol(T* X, T* W, T* B,
//  const unsigned int xh, const unsigned int xw,
//  const unsigned int wh, const unsigned int ww) 
//{
//  const unsigned int bh = xh;
//  const unsigned int bw = xw;
//  const unsigned int yh = xh;
//  const unsigned int yw = xw;
//
//  T* WX = mat2DMul(W, X, ww, wh, xw, xh);
//  T* Y = mat2DAdd(WX, B, yh, yw);
//  delete[] WX;
//  return Y;
//}
//
//// Y=W*X+B
//// Y,X,B都是行向量
//template<typename T>
//vector<vector<T>> fc2DRow(vector<vector<T>>& X,
//  vector<vector<T>>& W_, vector<vector<T>>& B) 
//{
//  vector<vector<T>> W = mat2DTranspose(W_);
//  const unsigned int hx = X.size();
//  const unsigned int wx = X[0].size();
//  const unsigned int hw = W.size();
//  const unsigned int ww = W[0].size();
//  const unsigned int hb = B.size();
//  const unsigned int wb = B[0].size();
//
//  vector<vector<T>> Y;
//  Y = mat2DMul(X, W);
//  Y = mat2DAdd(Y, B);
//  return Y;
//}
//template<typename T>
//T* fc2DRow(T* X, T* W_, T* B,
//  const unsigned int xh, const unsigned int xw,
//  const unsigned int wh_, const unsigned int ww_)
//{
//  T* W = mat2DTranspose(W_, wh_, ww_,NULL,NULL);
//  const unsigned int wh = ww_;
//  const unsigned int ww = wh_;
//  const unsigned int bh = xh;
//  const unsigned int bw = xw;
//  const unsigned int yh = xh;
//  const unsigned int yw = xw;
//
//  T* XW = mat2DMul(X, W, xw, xh, ww, wh);
//  T* Y = mat2DAdd(XW, B, bh, bw);
//
//  delete[] XW;
//  delete[] W;
//  return Y;
//}
//
//template<typename T>
//vector<vector<T>>  fc2D(vector<vector<T>>& X_,
//  vector<vector<T>>& W_, vector<vector<T>>& B_) 
//{
//  if (X_.size() == 1 &&B_.size()==1) {
//    // 行向量
//    //printf("ROW\n");
//    return fc2DRow(X_, W_, B_);
//  }
//  else if (X_[0].size() == 1&&B_[0].size()==1) {
//    // 列向量
//    //printf("COL\n");
//    return fc2DCol(X_, W_, B_);
//  }
//  else {
//    // UNDEFINE
//    assert(!"UNDEFINE IN FC2D");
//    return vector<vector<T>>(0);
//  }
//  
//}
//template<typename T>
//T* fc2D(T* X_, T* W_, T* B_,
//  const unsigned int xh, const unsigned int xw,
//  const unsigned int wh, const unsigned int ww,
//  const unsigned int bh, const unsigned int bw) 
//{
//  if (xh == 1 && bh == 1) {
//    // 行向量
//    printf("ROW\n");
//    return fc2DRow(X_, W_, B_, xh, xw, wh, ww);
//  }
//  else if (xw == 1 && bw == 1) {
//    // 列向量
//    printf("COL\n");
//    return fc2DCol(X_, W_, B_,xh,xw,wh,ww);
//  }
//  else {
//    // UNDEFINE
//    assert(!"UNDEFINE IN FC2D");
//    return vector<vector<T>>(0);
//  }
//}


template<typename T>
vector<T> fc2DMul(vector<T>& X, vector<vector<T>>& W)
{
  const unsigned int w = X.size();
  const unsigned int h = W.size();
  vector<T> res(h,0);
  for (unsigned int h_ = 0; h_ < h; ++h_) {
    for (unsigned int w_ = 0; w_ < w; ++w_) {
      res[h_] += W[h_][w_] * X[w_];
    }
  }
  return res;
}

template<typename T>
vector<T> fc2DAdd(vector<T>& X, vector<T>& B)
{
  const unsigned int w = X.size();
  vector<T> res(w);
  for (unsigned int w_ = 0; w_ < w; ++w_) {
    res[w_] = X[w_] + B[w_];
  }
  return res;
}


template<typename T>
vector<T> fc2D(vector<T>&X,
  vector<vector<T>>&W,vector<T>& B)
{
  vector<T> Y;

  // W*X
  Y = fc2DMul(X, W);

  // W*X+B
  Y = fc2DAdd(Y, B);
  return Y;
}






#endif