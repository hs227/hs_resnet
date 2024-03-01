
#ifndef INFER_H
#define INFER_H

#include"normal.h"


#include<time.h>
#include<stdlib.h>
#include<cmath>



template<typename T>
vector<T> relu(vector<T>& data)
{
  for_each(data.begin(), data.end(), [&](T& tmp) {
    tmp = max<T>(tmp, 0);
    });
  return data;
}
template<typename T>
T* relu(T* data, const unsigned int len)
{
  for (unsigned int i = 0; i < len; ++i) {
    data[i] = max(data[i], 0);
  }
  return data;
}

template<typename T>
vector<T> add(vector<T>& data1, vector<T>& data2)
{
  vector<T> sum;
  sum.resize(data1.size());
  transform(data1.begin(), data1.end(), data2.begin(), sum.begin(), std::plus<T>());
  return sum;
}
template<typename T>
T* add(T* data1, T* data2, const unsigned int len)
{
  T* sum = new T[len];
  for (unsigned int i = 0; i < len; ++i) {
    sum[i] = data1[i] + data2[i];
  }
  return sum;
}


template<typename T>
class Pic {
public:
  vector<vector<vector<T>>> data;// CHW
};

enum DSInit {
  DS_ZERO=0,
  DS_RAND,
  DS_SEQ,
};


template<typename T>
class DataStream {
public:
  vector<Pic<T>> datas;// NCHW
  unsigned int n;
  unsigned int width;
  unsigned int height;
  unsigned int channel;
public:
  DataStream() {
    //datas;
    n = 0;
    width = 0;
    height = 0;
    channel = 0;
  }
  DataStream(const unsigned int n_, const unsigned int w_,
    const unsigned int h_, const unsigned int c_,enum DSInit flag= DS_ZERO)
  {
    srand(time(0));
    // [NWHC]
    n = n_;
    width = w_;
    height = h_;
    channel = c_;
    //n = (rand() % 3) + 1;// 1~3
    //channel = (rand() % 3) + 1;// 1~3
    datas.resize(n);
    for_each(datas.begin(), datas.end(), [&](Pic<T>& pic) {
      pic.data = vector<vector<vector<T>>>(channel,
      vector<vector<T>>(height, vector<T>(width,0)));
      });

    //printf("n=%d,c=%d,h=%d,w=%d\n",
    //  datas.size(),datas[0].data.size(),
    //  datas[0].data[0].size(),datas[0].data[0][0].size());
    if (flag == DS_RAND) {
      for_each(datas.begin(), datas.end(), [&](Pic<T>& pic) {
        for (unsigned int ic = 0; ic < channel; ++ic) {
          for (unsigned int ih = 0; ih < height; ++ih) {
            for (unsigned int iw = 0; iw < width; ++iw) {
              pic.data[ic][ih][iw] = (float)(rand() % 256);
            }
          }
        }
        });
    }
    else if (flag == DS_SEQ) {
      for_each(datas.begin(), datas.end(), [&](Pic<T>& pic) {
        for (unsigned int ic = 0; ic < channel; ++ic) {
          for (unsigned int ih = 0; ih < height; ++ih) {
            for (unsigned int iw = 0; iw < width; ++iw) {
              pic.data[ic][ih][iw] = iw;
            }
          }
        }
        });
    
    
    }
  }
  DataStream(DataStream<T>& old) :
    DataStream(old.n, old.width, old.height,old.channel)
  {}
  DataStream(vector<T*> tdatas,
    const unsigned int w_, const unsigned int h_, const unsigned int c_) 
  {
    n = tdatas.size();
    width = w_;
    height = h_;
    channel = c_;

    datas.resize(n);
    for_each(datas.begin(), datas.end(), [&](Pic<T>& pic) {
      pic.data = vector<vector<vector<T>>>(channel,
      vector<vector<T>>(height, vector<T>(width, 0)));
      });

    for (unsigned int in = 0; in < n;++in) {
      for (unsigned int ic = 0; ic < channel; ++ic) {
        for (unsigned int ih = 0; ih < height; ++ih) {
          for (unsigned int iw = 0; iw < width; ++iw) {
            Pic<T>& pic = datas[in];
            pic.data[ic][ih][iw] = tdatas[in][ic*height*width+ih*width+iw];
          }
        }
      }
      }



  }


  ~DataStream()
  {}


  vector<T*> vout(unsigned int*n_,unsigned int* h_,
    unsigned int*w_,unsigned int*c_) 
  {
    if (n_) 
      *n_ = n;
    if (h_)
      *h_ = height;
    if (w_)
      *w_ = width;
    if (c_)
      *c_ = channel;
    vector<T*> outputs;
    
    for (unsigned int in = 0; in < n; ++in) {
      T* output = (T*)malloc(n * channel * height * width * sizeof(T));
      for (unsigned int ic = 0; ic < channel; ++ic) {
        for (unsigned int ih = 0; ih < height; ++ih) {
          memcpy(&output[ic * height * width + ih * width],
            &datas[in].data[ic][ih][0],
            sizeof(T) * width);
          //for (unsigned int iw = 0; iw < width; ++iw) {
          //  output[ic * height * width + ih * width + iw] =
          //    datas[in].data[ic][ih][iw];
          //}
        }
      }
      outputs.push_back(output);
    }

    return outputs;
  }



  T& getpixel(const unsigned int n, const unsigned int c,
    const unsigned int h, const unsigned int w) 
  {
    return datas[n].data[c][h][w];
  }

  Pic<T>& getpic(const unsigned int n) 
  {
    return datas[n];
  }

  vector<vector<T>>& getpicc(const unsigned int n, const unsigned int c) 
  {
    return datas[n].data[c];
  }

  void ShowTravel(void) 
  {
    printf("n=%zd,c=%zd,h=%zd,w=%zd\n",
      datas.size(),datas[0].data.size(),
      datas[0].data[0].size(),datas[0].data[0][0].size());
    std::cout << "DataStream Showing" << std::endl;
    for (int in = 0; in < n; ++in) {
      std::cout << "N=" << in <<" :"<< std::endl;
      for (int ic = 0; ic < channel; ++ic) {
        std::cout << " C=" << ic << " :" << std::endl;
        std::cout << "[" << std::endl;
        for (int ih = 0; ih < height; ++ih) {
          std::cout << " [" ;
          for (int iw = 0; iw < width; ++iw) {
            T tmp = datas[in].data[ic][ih][iw];
            std::cout << ", " << tmp;
          }
          std::cout << " ]" << std::endl;
        }
        std::cout << "]" << std::endl;
      }
    }
    std::cout << "Showing End" << std::endl;
  }

};











#endif
