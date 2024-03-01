#pragma once
#ifndef POOLING_H
#define POOLING_H

#include"normal.h"



enum POOLTYPE {
  MAXPOOLING = 0,
  AVEPOOLING,
  POOLTYPE_LIMIT,
};





template<typename T>
T pool2DMax(T* matx, const unsigned int width, const unsigned int height)
{
  T res = std::numeric_limits<T>::min();
  for (unsigned int y = 0; y < height; ++y) {
    for (unsigned int x = 0; x < width; ++x) {
      T tmp = mat2DWindowElem<T>(matx, width, height, 0, 0, x, y);
      if (tmp > res)
        res = tmp;
    }
  }
  return res;
}
template<typename T>
T pool2DMax(vector<vector<T>>& matx)
{
  const unsigned int height = matx.size();
  const unsigned int width = matx[0].size();
  T res = std::numeric_limits<T>::min();

  for (unsigned int y = 0; y < height; ++y) {
    for (unsigned int x = 0; x < width; ++x) {
      T tmp = mat2DWindowElem(matx, 0, 0, x, y);
      if (tmp > res)
        res = tmp;
    }
  }
  return res;
}

template<typename T>
T pool2DAve(T* matx, const unsigned int width, const unsigned int height)
{
  T res = 0;
  for (unsigned int y = 0; y < height; ++y) {
    for (unsigned int x = 0; x < width; ++x) {
      T tmp = mat2DWindowElem<T>(matx, width, height, 0, 0, x, y);
      res += tmp;
    }
  }
  res /= width * height;
  return res;
}
template<typename T>
T pool2DAve(vector<vector<T>>& matx)
{
  const unsigned int height = matx.size();
  const unsigned int width = matx[0].size();
  T res = 0;

  for (unsigned int y = 0; y < height; ++y) {
    for (unsigned int x = 0; x < width; ++x) {
      T tmp = mat2DWindowElem(matx, 0, 0, x, y);
      res += tmp;
    }
  }
  res /= width * height;
  return res;
}



template<typename T>
T* pool2D(T* inMatx_, const unsigned int in_width_, const unsigned int in_height_,
  const unsigned int k_width, const unsigned int k_height,
  const unsigned int padding[4], const unsigned int stride[2],
  unsigned int* const out_width_, unsigned int* const out_height_,
  enum POOLTYPE flag)
{
  if (flag >= POOLTYPE_LIMIT)
    return nullptr;

  /* padding */
  unsigned int in_width, in_height;
  T* inMatx = mat2DPadding<T>((T*)inMatx_, in_width_, in_height_,
    padding[0], padding[1], padding[2], padding[3], &in_width, &in_height);

  const unsigned int stride_height = stride[0];
  const unsigned int stride_width = stride[1];

  T* outMatx = nullptr;
  const unsigned int out_width = (in_width - k_width) / stride_width + 1;
  const unsigned int out_height = (in_height - k_height) / stride_height + 1;
  if (out_width_ != nullptr)
    *out_width_ = out_width;
  if (out_height_ != nullptr)
    *out_height_ = out_height;
  outMatx = new T[out_width * out_height];
  T* window = new T[k_width * k_height];

  for (unsigned int y = 0; y < out_height; ++y) {
    for (unsigned int x = 0; x < out_width; ++x) {
      mat2DWindow<T>(inMatx, in_width, in_height, window, x * stride_width, y * stride_height, k_width, k_height);
      T res;
      if (flag == POOLTYPE::MAXPOOLING) {
        res = pool2DMax<T>(window, k_width, k_height);
      }
      else if (flag == POOLTYPE::AVEPOOLING) {
        res = pool2DAve<T>(window, k_width, k_height);
      }
      mat2DWindowElem(outMatx, out_width, out_height, 0, 0, x, y) = res;
    }
  }

  delete[] window;
  delete[] inMatx;

  return outMatx;
}
template<typename T>
vector<vector<T>> pool2D(vector<vector<T>>& inMatx_,
  const unsigned int k_width, const unsigned int k_height,
  const unsigned int padding[4], const unsigned int stride[2],
  enum POOLTYPE flag)
{
  if (flag >= POOLTYPE_LIMIT)
    assert(!"pool2D POOLTYPE_LIMIT::ERROR");
  /* padding */
  vector<vector<T>> inMatx = mat2DPadding(inMatx_,
    padding[0], padding[1], padding[2], padding[3]);

  unsigned int in_width = inMatx[0].size();
  unsigned int in_height = inMatx.size();
  const unsigned int stride_height = stride[0];
  const unsigned int stride_width = stride[1];
  const unsigned int out_width = (in_width - k_width) / stride_width + 1;
  const unsigned int out_height = (in_height - k_height) / stride_height + 1;

  vector<vector<T>> outMatx(out_height, vector<T>(out_width));
  vector<vector<T>> window(k_height, vector<T>(k_width));

  for (unsigned int y = 0; y < out_height; ++y) {
    for (unsigned int x = 0; x < out_width; ++x) {
      mat2DWindow(inMatx, window, x * stride_width, y * stride_height, k_width, k_height);
      T res;
      if (flag == POOLTYPE::MAXPOOLING) {
        res = pool2DMax(window);
      }
      else if (flag == POOLTYPE::AVEPOOLING) {
        res = pool2DAve(window);
      }
      mat2DWindowElem(outMatx, 0, 0, x, y) = res;
    }
  }
  return outMatx;
}


#endif