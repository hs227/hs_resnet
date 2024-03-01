
#ifndef NORMAL_H
#define NORMAL_H

#include<iostream>
#include<vector>
#include<cmath>

#include <algorithm>
#include <numeric>
#include<cassert>
using namespace std;

template<typename T>
void printMat2D(T* matx, const unsigned int lenX, const unsigned int lenY)
{
  for (int y = 0; y < lenY; ++y) {
    for (int x = 0; x < lenX; ++x) {
      std::cout << *(matx + y * lenX + x) << " ";
      //printf("%2.0f ", *(matx + y * lenX + x));
    }
    std::cout << std::endl;
  }
}
template<typename T>
void printMat2D(vector<vector<T>>& matx)
{
  std::cout << "[" << std::endl;
  for_each(matx.begin(), matx.end(), [&](vector<T>& one_row) {
    std::cout << "[";
    for_each(one_row.begin(), one_row.end(), [&](T& data) {
      std::cout << ", " << data;
      });
    std::cout << "]" << std::endl;
    });
  std::cout << "]" << std::endl;
}

template<typename T>
T& mat2DWindowElem(T* matx, const unsigned int realX, const unsigned int realY,
  const unsigned int windowX, const unsigned int windowY,
  const unsigned int wIdxX, const unsigned int wIdxY)
{
  const unsigned int x = windowX + wIdxX;
  const unsigned int y = windowY + wIdxY;
  //if (x > realX || y > realY)
  //  throw std::runtime_error("<mat2DWindow>error1");
  return *(matx + y * realX + x);
}
template<typename T>
T& mat2DWindowElem(vector<vector<T>>& matx,
  const unsigned int windowX, const unsigned int windowY,
  const unsigned int wIdxX, const unsigned int wIdxY)
{
  const unsigned int x = windowX + wIdxX;
  const unsigned int y = windowY + wIdxY;
  return matx[y][x];
}

template<typename T>
T* mat2DWindow(T* matx, const unsigned int realX, const unsigned int realY,
  T* wMatx, const unsigned int windowX, const unsigned int windowY,
  const unsigned int wIdxX, const unsigned int wIdxY)
{
  const unsigned int x = windowX + wIdxX;
  const unsigned int y = windowY + wIdxY;
  //if (x > realX || y > realY)
  //  return nullptr;
  //T* matx = (matx_ + windowY * realX + windowX);
  for (int iy = 0; iy < wIdxY; ++iy) {
    for (int ix = 0; ix < wIdxY; ++ix) {
      *(wMatx + iy * wIdxX + ix) = mat2DWindowElem(matx, realX, realY, windowX, windowY, ix, iy);
    }
  }
  return wMatx;
}
template<typename T>
vector<vector<T>>& mat2DWindow(vector<vector<T>>& matx, vector<vector<T>>& wMatx,
  const unsigned int windowX, const unsigned int windowY,
  const unsigned int wIdxX, const unsigned int wIdxY)
{
  const unsigned int x = windowX + wIdxX;
  const unsigned int y = windowY + wIdxY;

  for (int iy = 0; iy < wIdxY; ++iy) {
    for (int ix = 0; ix < wIdxX; ++ix) {
      wMatx[iy][ix] = mat2DWindowElem(matx, windowX, windowY, ix, iy);
    }
  }
  return wMatx;
}


template<typename T>
T* mat2DWindowT(T* matx, const unsigned int realX, const unsigned int realY,
  T* wMatx, const unsigned int windowX, const unsigned int windowY,
  const unsigned int wIdxX, const unsigned int wIdxY)
{
  const unsigned int x = windowX + wIdxX;
  const unsigned int y = windowY + wIdxY;
  for (unsigned int iy = 0; iy < wIdxY; ++iy) {
    for (unsigned int ix = 0; ix < wIdxY; ++ix) {
      mat2DWindowElem(matx, realX, realY, windowX, windowY, ix, iy) = *(wMatx + iy * wIdxX + ix);
    }
  }
  return wMatx;
}
template<typename T>
vector<vector<T>>& mat2DWindowT(vector<vector<T>>& matx, vector<vector<T>>& wMatx,
  const unsigned int windowX, const unsigned int windowY,
  const unsigned int wIdxX, const unsigned int wIdxY)
{
  const unsigned int x = windowX + wIdxX;
  const unsigned int y = windowY + wIdxY;
  for (unsigned int iy = 0; iy < wIdxY; ++iy) {
    for (unsigned int ix = 0; ix < wIdxX; ++ix) {
      mat2DWindowElem(matx, windowX, windowY, ix, iy) = wMatx[iy][ix];
    }
  }
  return wMatx;
}

enum TraversalFlag {
  PRINT = 0,
  BIG,
};

template<typename T>
T* convTraversal(T* matx, const unsigned int realX, const unsigned int realY,
  const unsigned int wIdxX, const unsigned int wIdxY, enum TraversalFlag flag)
{
  T* window = new T[wIdxY * wIdxX];
  unsigned int turnY = realY - wIdxY + 1;
  unsigned int turnX = realX - wIdxX + 1;
  T* bigWindow = nullptr;
  if (flag == TraversalFlag::BIG)
    bigWindow = new T[(turnY * wIdxY) * (turnX * wIdxX)];

  for (int y = 0; y < turnY; y++) {
    for (int x = 0; x < turnX; x++) {
      mat2DWindow<T>(matx, realX, realY, window, x, y, wIdxX, wIdxY);
      if (flag == TraversalFlag::PRINT) {
        std::cout << "[" << y << "," << x << "]" << std::endl;
        printMat2D<T>(window, wIdxX, wIdxY);
        std::cout << std::endl;
      }
      else if (flag == TraversalFlag::BIG) {
        unsigned int iy = y * wIdxY;
        unsigned int ix = x * wIdxX;
        //mat2DWindowT<T>(window, wIdxX, wIdxY, (bigWindow+iy*(turnX*wIdxX)+ix), 0, 0, wIdxX, wIdxY);
        mat2DWindowT<T>(bigWindow, turnX * wIdxX, turnY * wIdxY, window, ix, iy, wIdxX, wIdxY);
      }
    }
  }
  delete[] window;
  if (flag == TraversalFlag::BIG)
    return bigWindow;
  return nullptr;
}
template<typename T>
vector<vector<T>> convTraversal(vector<vector<T>> matx,
  const unsigned int wIdxX, const unsigned int wIdxY, enum TraversalFlag flag)
{
  vector<vector<T>> window(wIdxY, vector<T>(wIdxX));
  const unsigned int realY = matx.size();
  const unsigned int realX = matx[0].size();
  unsigned int turnY = realY - wIdxY + 1;
  unsigned int turnX = realX - wIdxX + 1;
  vector<vector<T>> bigWindow;
  if (flag == TraversalFlag::BIG)
    bigWindow = vector<vector<T>>(turnY * wIdxY, vector<T>(turnX * wIdxX));

  for (int y = 0; y < turnY; y++) {
    for (int x = 0; x < turnX; x++) {
      mat2DWindow(matx, window, x, y, wIdxX, wIdxY);
      if (flag == TraversalFlag::PRINT) {
        std::cout << "[" << y << "," << x << "]" << std::endl;
        printMat2D(window);
        std::cout << std::endl;
      }
      else if (flag == TraversalFlag::BIG) {
        unsigned int iy = y * wIdxY;
        unsigned int ix = x * wIdxX;
        mat2DWindowT(bigWindow, window, ix, iy, wIdxX, wIdxY);
      }
    }
  }

  return bigWindow;
}

template<typename T>
T* mat2DPadding(T* matx, const unsigned int width, const unsigned int height,
  const unsigned int pad_up, const unsigned int pad_down,
  const unsigned int pad_left, const unsigned int pad_right,
  unsigned int* const new_width_, unsigned int* const new_height_)
{
  unsigned int new_width = width + pad_left + pad_right;
  unsigned int new_height = height + pad_up + pad_down;
  if (new_width_ != nullptr)
    *new_width_ = new_width;
  if (new_height_ != nullptr)
    *new_height_ = new_height;
  T* new_matx = new T[new_width * new_height];
  for (unsigned int y = 0; y < new_height; ++y) {
    for (unsigned int x = 0; x < new_width; ++x) {
      mat2DWindowElem(new_matx, new_width, new_height, 0, 0, x, y) = 0;
    }
  }
  mat2DWindowT(new_matx, new_width, new_height, matx, pad_left, pad_up, width, height);
  return new_matx;
}
template<typename T>
vector<vector<T>> mat2DPadding(vector<vector<T>>& matx,
  const unsigned int pad_up, const unsigned int pad_down,
  const unsigned int pad_left, const unsigned int pad_right)
{
  unsigned int width = matx[0].size();
  unsigned int height = matx.size();
  unsigned int new_width = width + pad_left + pad_right;
  unsigned int new_height = height + pad_up + pad_down;

  vector<vector<T>>new_matx(new_height, vector<T>(new_width, 0));
  mat2DWindowT(new_matx, matx, pad_left, pad_up, width, height);
  
  return new_matx;
}


template<typename T>
void printVec(vector<T>& data)
{
  std::cout << "[";
  for_each(data.begin(), data.end(), [&](T tmp) {
    std::cout << ", " << tmp;
    });
  std::cout << "]" << std::endl;
}
template<typename T>
void printVec(T* data, const unsigned int len)
{
  std::cout << "[";
  for (unsigned int i = 0; i < len; ++i) {
    std::cout << ", " << data[i];
  }
  std::cout << "]" << std::endl;
}



template<typename T>
T* vvt2t(vector<vector<T>>& raw_data,T**new_data) 
{
  const unsigned int height = raw_data.size();
  const unsigned int width = raw_data[0].size();

  *new_data = new T[width * height];
  for (unsigned int h = 0; h < height; ++h) {
    for (unsigned int w = 0; w < width; ++w) {
      *(*new_data + h * width+w) = raw_data[h][w];
    }
  }
  return *new_data;
}
template<typename T>
vector<vector<T>>& t2vvt(T* raw_data,vector<vector<T>>& new_data, 
  const unsigned int width, const unsigned int height)
{
  new_data=vector<vector<T>>(height, vector<T>(height));
  for (unsigned int h = 0; h < height; ++h) {
    for (unsigned int w = 0; w < width; ++w) {
      new_data[h][w] = *(raw_data + h * width + w);
    }
  }
  return new_data;
}

template<typename T>
vector<vector<T>> vvtplus(vector<vector<T>>& vvt1,
  vector<vector<T>>& vvt2,
  vector<vector<T>>& vvt3) 
{
  const unsigned int height = vvt1.size();
  const unsigned int width = vvt1[0].size();

  for (unsigned int ih = 0; ih < height; ++ih) {
    for (unsigned int iw = 0; iw < width; ++iw) {
      vvt3[ih][iw] = vvt1[ih][iw] + vvt2[ih][iw];
    }
  }
  return vvt3;
}


template<typename T>
vector<vector<T>> mat2DTranspose(vector<vector<T>>& input)
{
  const unsigned int ih = input.size();
  const unsigned int iw = input[0].size();
  const unsigned int oh = iw;
  const unsigned int ow = ih;

  vector<vector<T>> output(oh, vector<T>(ow));

  for (unsigned int oh_ = 0; oh_ < oh; ++oh_) {
    for (unsigned int ow_ = 0; ow_ < ow; ++ow_) {
      output[oh_][ow_] = input[ow_][oh_];
    }
  }
  return output;
}
template<typename T>
T* mat2DTranspose(T* input, const unsigned int ih, const unsigned int iw,
  unsigned int* const oh_, unsigned int* const ow_)
{
  const unsigned int oh = iw;
  const unsigned int ow = ih;
  if (oh_)
    *oh_ = oh;
  if (ow_)
    *ow_ = ow;

  T* output = new T[oh * ow];
  for (unsigned int ioh = 0; ioh < oh; ++ioh) {
    for (unsigned int iow = 0; iow < ow; ++iow) {
      //output[ioh][iow] = input[iow][ioh];
      *(output + ioh * ow + iow) = *(input + iow * iw + ioh);
    }
  }
  return output;
}



template<typename T>
T mat2DHWMul(vector<vector<T>>& data1, vector<vector<T>>& data2,
  const unsigned int oh, const unsigned int ow)
{
  vector<vector<T>>data2T = mat2DTranspose(data2);
  vector<T> sums(data1[0].size());
  transform(data1[oh].begin(), data1[oh].end(), data2T[ow].begin(), sums.begin(), std::multiplies<T>());
  T res = accumulate(sums.begin(), sums.end(), 0);
  return res;
}
template<typename T>
T mat2DHWMul(T* data1, T* data2,
  const unsigned int w1, const unsigned int h1,
  const unsigned int w2, const unsigned int h2,
  const unsigned int oh, const unsigned int ow)
{
  T sums = 0;
  for (unsigned int w_ = 0; w_ < w1; ++w_) {
    sums += mat2DWindowElem(data1, w1, h1, 0, 0, w_, oh) *
      mat2DWindowElem(data2, w2, h2, 0, 0, ow, w_);
  }
  return sums;
}

template<typename T>
vector<vector<T>> mat2DMul(vector<vector<T>>& data1, vector<vector<T>>& data2)
{
  const unsigned int h1 = data1.size();
  const unsigned int w1 = data1[0].size();
  const unsigned int h2 = data2.size();
  const unsigned int w2 = data2[0].size();

  if (w1 != h2) {
    assert(!"w1!=h2");
    return vector<vector<T>>(0);
  }
  const unsigned int ho = h1;
  const unsigned int wo = w2;
  vector<vector<T>> output(ho, vector<T>(wo));

  for (unsigned int ho_ = 0; ho_ < ho; ++ho_) {
    for (unsigned int wo_ = 0; wo_ < wo; ++wo_) {
      output[ho_][wo_] = mat2DHWMul(data1, data2, ho_, wo_);
    }
  }
  return output;
}
template<typename T>
T* mat2DMul(T* data1, T* data2,
  const unsigned int w1, const unsigned int h1,
  const unsigned int w2, const unsigned int h2)
{
  if (w1 != h2) {
    assert(!"w1!=h2");
    return nullptr;
  }
  const unsigned int ho = h1;
  const unsigned int wo = w2;

  T* output = new T[ho * wo];

  for (unsigned int ho_ = 0; ho_ < ho; ++ho_) {
    for (unsigned int wo_ = 0; wo_ < wo; ++wo_) {
      mat2DWindowElem(output, wo, ho, 0, 0, wo_, ho_) = mat2DHWMul(
        data1, data2, w1, h1, w2, h2, ho_, wo_);
    }
  }

  return output;
}




template<typename T>
vector<vector<T>> mat2DAdd(vector<vector<T>>& m1,
  vector<vector<T>>& m2)
{
  const unsigned int h = m1.size();
  const unsigned int w = m1[0].size();
  vector<vector<T>> out(h, vector<T>(w));
  for (unsigned int h_ = 0; h_ < h; ++h_) {
    for (unsigned int w_ = 0; w_ < w; ++w_) {
      out[h_][w_] = m1[h_][w_] + m2[h_][w_];
    }
  }
  return out;
}
template<typename T>
T* mat2DAdd(T* m1, T* m2, const unsigned int h, const unsigned int w)
{
  T* out = new T[h * w];
  for (unsigned int h_ = 0; h_ < h; ++h_) {
    for (unsigned int w_ = 0; w_ < w; ++w_) {
      mat2DWindowElem(out, w, h, 0, 0, w_, h_) = mat2DWindowElem(m1, w, h, 0, 0, w_, h_) +
        mat2DWindowElem(m2, w, h, 0, 0, w_, h_);
    }
  }
  return out;
}



#endif