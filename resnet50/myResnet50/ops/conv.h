
#ifndef CONV_H
#define CONV_H

#include"normal.h"



template<typename T>
T conv2DMul(T* matx1, T* matx2, const unsigned int lenX, const unsigned int lenY)
{
  T res = 0;
  for (int y = 0; y < lenY; ++y) {
    for (int x = 0; x < lenX; ++x) {
      res += (*(matx1 + y * lenX + x)) * (*(matx2 + y * lenX + x));
    }
  }
  return res;
}
template<typename T>
T conv2DMul(vector<vector<T>>& matx1, vector<vector<T>>& matx2)
{
  T res = 0;
  unsigned int lenY = matx1.size();
  unsigned int lenX = matx1[0].size();
  for (int y = 0; y < lenY; ++y) {
    for (int x = 0; x < lenX; ++x) {
      res += matx1[y][x] * matx2[y][x];
    }
  }
  return res;
}

template<typename T>
T* conv2D(T* inMatx_, const unsigned int in_width_, const unsigned int in_height_,
  T* wMatx, const unsigned int w_width, const unsigned int w_height,
  const unsigned int padding[4], const unsigned int stride[2],
  unsigned int* const out_width_, unsigned int* const out_height_)
{
  /* padding */
  unsigned int in_width, in_height;
  T* inMatx = mat2DPadding<T>((T*)inMatx_, in_width_, in_height_,
    padding[0], padding[1], padding[2], padding[3], &in_width, &in_height);

  const unsigned int stride_height = stride[0];
  const unsigned int stride_width = stride[1];

  T* outMatx = nullptr;
  const unsigned int out_width = (in_width - w_width) / stride_width + 1;
  const unsigned int out_height = (in_height - w_height) / stride_height + 1;
  if (out_width_ != nullptr)
    *out_width_ = out_width;
  if (out_height_ != nullptr)
    *out_height_ = out_height;
  outMatx = new T[out_width * out_height];
  T* window = new T[w_width * w_height];

  for (unsigned int y = 0; y < out_height; ++y) {
    for (unsigned int x = 0; x < out_width; ++x) {
      mat2DWindow<T>(inMatx, in_width, in_height, window, x * stride_width, y * stride_height, w_width, w_height);
      T convRes = conv2DMul(window, wMatx, w_width, w_height);
      mat2DWindowElem(outMatx, out_width, out_height, 0, 0, x, y) = convRes;
    }
  }

  delete[] window;
  delete[] inMatx;

  return outMatx;
}
template<typename T>
vector<vector<T>> conv2D(vector<vector<T>>& inMatx_, vector<vector<T>>& wMatx,
  const unsigned int padding[4], const unsigned int stride[2])
{
  /* padding */
  vector<vector<T>> inMatx = mat2DPadding(inMatx_,
    padding[0], padding[1], padding[2], padding[3]);

  unsigned int in_width = inMatx[0].size();
  unsigned int in_height = inMatx.size();
  unsigned int w_width = wMatx[0].size();
  unsigned int w_height = wMatx.size();
  const unsigned int stride_height = stride[0];
  const unsigned int stride_width = stride[1];
  const unsigned int out_width = (in_width - w_width) / stride_width + 1;
  const unsigned int out_height = (in_height - w_height) / stride_height + 1;

  vector<vector<T>> outMatx(out_height, vector<T>(out_width));
  vector<vector<T>> window(w_height, vector<T>(w_width));

  for (unsigned int y = 0; y < out_height; ++y) {
    for (unsigned int x = 0; x < out_width; ++x) {
      mat2DWindow(inMatx, window, x * stride_width, y * stride_height, w_width, w_height);
      T convRes = conv2DMul(window, wMatx);
      mat2DWindowElem(outMatx, 0, 0, x, y) = convRes;
    }
  }
  return outMatx;
}


#endif