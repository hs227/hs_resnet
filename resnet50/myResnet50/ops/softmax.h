#pragma once
#ifndef SOFTMAX_H
#define SOFTMAX_H

#include"normal.h"


template<typename T>
vector<double> softmax(vector<T>& data)
{
  const unsigned int len = data.size();
  vector<double> probs(len,0);
  double sums = 0;
  for (unsigned int i = 0; i < len; ++i) {
    double ex = exp(data[i]);
    probs[i] = ex;
    sums += ex;
  }
  for (unsigned int i = 0; i < len; ++i) {
    probs[i] /= sums ;

  }
  return probs;
}

template<typename T>
double* softmax(T* data, const unsigned int len) 
{
  double* probs = new double[len];
  double sums = 0;
  for (unsigned int i = 0; i < len; ++i) {
    double ex = exp(data);
    probs[i] = ex;
    sums += ex;
  }
  for (unsigned int i = 0; i < len; ++i) {
    probs[i] /= sums;
  }
  return probs;
}


template<typename T>
unsigned int findMax(vector<T> data) 
{
  unsigned int max = 0;
  for (unsigned int i = 1; i < data.size(); ++i) 
  {
    if (data[i] > data[max])
      max = i;
  }
  return max;
}
template<typename T>
unsigned int findMax(T* data, const unsigned int len) 
{
  unsigned int max = 0;
  for (unsigned int i = 1; i < len; ++i) {
    if (data[i] > data[max])
      max = i;
  }
  return max;
}





#endif