#pragma once
#ifndef BN_H
#define BN_H

#include"normal.h"

template<typename T>
pair<vector<T>, vector<T>> meanAndStd(vector<vector<T>>& data)
{
  const T no_zero = 1e-8;
  // mean
  vector<T> means(data[0].size(), 0);
  for_each(data.begin(), data.end(), [&](vector<T> batch_data) {
    //printVec(batch_data);
    transform(means.begin(), means.end(), batch_data.begin(), means.begin(), std::plus<T>());
    });
  for_each(means.begin(), means.end(), [&](T& tmp) {
    tmp /= data.size();
    });
  //printf("means:");
  //printVec(means);
  //return means;

  // variance
  vector<T> vars(means.size(), 0);
  for_each(data.begin(), data.end(), [&](vector<T> batch_data) {
    vector<T> minus(means.size(), 0);
    transform(means.begin(), means.end(), batch_data.begin(), minus.begin(), std::minus<T>());
    //printVec(minus);
    transform(minus.begin(), minus.end(), minus.begin(), minus.begin(), std::multiplies<T>());
    transform(minus.begin(), minus.end(), vars.begin(), vars.begin(), std::plus<T>());
    });
  //printf("vars:");
  //printVec(vars);
  //return vars;
  // stdd
  for_each(vars.begin(), vars.end(), [&](T& tmp) {
    tmp /= (data.size()+no_zero);
    tmp = sqrt(tmp);
    });
  //printf("std:");
  //printVec(vars);

  return pair<vector<T>, vector<T>>{means, vars};
}
template<typename T>
pair<T*, T*> meanAndStd(T* data, const unsigned int lenX, const unsigned int lenY)
{
  const T no_zero = 1e-8;
  // mean
  T* means = new T[lenX];
  for (unsigned int x = 0; x < lenX; ++x) {
    means[x] = 0;
    for (unsigned int y = 0; y < lenY; ++y) {
      means[x] += *(data + y * lenX + x);
    }
    means[x] /= lenY;
  }
  //printf("means:");
  //printVec(means, lenX);
  //return means;


  // variance
  T* vars = new T[lenX];
  T* minus_tmp = new T[lenX];
  for (unsigned int x = 0; x < lenX; ++x) {
    vars[x] = 0;
    minus_tmp[x] = 0;
    for (unsigned int y = 0; y < lenY; ++y) {
      minus_tmp[x] = *(data + y * lenX + x) - means[x];
      minus_tmp[x] = pow(minus_tmp[x], 2);
      vars[x] += minus_tmp[x];
    }
  }
  delete[] minus_tmp;
  //printf("vars:");
  //printVec(vars, lenX);
  //return vars;

  // stdd
  for (unsigned int x = 0; x < lenX; ++x) {
    vars[x] /= (lenY+no_zero);
    vars[x] = sqrt(vars[x]);
  }
  //printf("std:");
  //printVec(vars, lenX);

  return pair<T*, T*>{means, vars};
}



template<typename T>
vector<vector<T>> batch_norm(vector<vector<T>>& datas, T gamma, T beta,
  T means,T stdds)
{
  T eps = 1e-5;
  //pair<vector<T>, vector<T>> pairs = meanAndStd<T>(datas);
  //means = pairs.first;
  //stdds = pairs.second;
  //printf("means:");
  //printVec(means);
  //printf("stdds:");
  //printVec(stdds);
  //return 0;

  vector<vector<T>>norms(datas);
  //printf("norms:[\n");

  for_each(norms.begin(), norms.end(), [&](vector<T>& batch_data) {
    for (int i = 0; i < batch_data.size(); ++i) {
      batch_data[i] = (batch_data[i] - means) / (stdds+eps);
      batch_data[i] = gamma * batch_data[i] + beta;
    }
    //printVec(batch_data);
    });
  //printf("]\n");

  return norms;
}
template<typename T>
T* batch_norm(T* datas, const unsigned int lenX, const unsigned int lenY,
  T gamma, T beta)
{
  pair<vector<T>, vector<T>> pairs = meanAndStd<T>(datas);
  vector<T> means = pairs.first;
  vector<T> stdds = pairs.second;
  //printf("means:");
  //printVec(means);
  //printf("stdds:");
  //printVec(stdds);
  //return 0;
  pair<T*, T*> pairs = meanAndStd<T>(datas, lenX, lenY);
  T* means = pairs.first;
  T* stdds = pairs.second;
  printf("means:");
  printVec(means, lenX);
  printf("stdds:");
  printVec(stdds, lenX);
  //return 0;

  vector<vector<T>>norms(datas);
  T* norms = new T[lenX * lenY];
  printf("norms:[\n");
  for (unsigned int y = 0; y < lenY; ++y) {
    for (unsigned int x = 0; x < lenX; ++x) {
      *(norms + y * lenX + x) = ((datas + y * lenX + x) - means[x]) / stdds[x];
      *(norms + y * lenX + x) = gamma * (*(norms + y * lenX + x)) + beta;
    }
    printVec(norms + y * lenX, lenX);
  }
  printf("]\n");

  return norms;
}


#endif

