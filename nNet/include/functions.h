#pragma once

#include<iostream>
#include<opencv2\opencv.hpp>
#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>

// sigmoid function
static cv::Mat sigmoid(cv::Mat& x)
{
  // 1/(1+exp(-x))
  cv::Mat exp_x;
  cv::exp(-x, exp_x);
  cv::Mat fx = 1.0 / (1.0 + exp_x);
  return fx;
}
static cv::Mat sigmoid_prime(cv::Mat& fx) 
{
  cv::Mat s = sigmoid(fx);
  cv::Mat dx = s * (1.0 - s);
  return dx;
}


// Tanh fucntion
static cv::Mat tanh(cv::Mat& x)
{
  // (e(x)-e(-x))/(e(x)+e(-x))
  cv::Mat exp_x, exp_mx;
  cv::exp(x, exp_x);
  cv::exp(-x, exp_mx);
  cv::Mat fx = (exp_x - exp_mx) / (exp_x + exp_mx);
  return fx;
}
static cv::Mat tanh_prime(cv::Mat& fx) 
{
  cv::Mat tanh_2;
  pow(tanh(fx), 2., tanh_2);
  cv::Mat dx = 1 - tanh_2;
  return dx;
}


// relu function
static cv::Mat relu(cv::Mat& x)
{
  cv::Mat fx = x.clone();
  for (int r_ = 0; r_ < fx.rows; ++r_) {
    for (int c_ = 0; c_ < fx.cols; ++c_) {
      if (fx.at<float>(r_, c_) < 0) {
        fx.at<float>(r_, c_) = 0;
      }
    }
  }
  return fx;
}
static cv::Mat relu_prime(cv::Mat& fx) 
{
  cv::Mat dx = fx.clone();
  for (int r_ = 0; r_ < dx.rows; ++r_) {
    for (int c_ = 0; c_ < dx.cols; ++c_) {
      if (dx.at<float>(r_, c_) > 0) {
        dx.at<float>(r_, c_) = 1;
      }
    }
  }
  return dx;
}

// 均方误差损失函数
static cv::Mat mseLoss(const cv::Mat& y_pred, const cv::Mat& y_true)
{
  cv::Mat minus = y_pred - y_true;
  cv::Mat y_pow2;
  pow(minus, 2, y_pow2);
  cv::Mat loss;
  cv::reduce(y_pow2, loss, 0, cv::REDUCE_AVG);
  return loss;
}
// 均方误差损失导数
static cv::Mat mseLoss_prime(const cv::Mat& y_pred, const cv::Mat& y_true) 
{
  cv::Mat minus = y_pred - y_true;
  cv::Mat res = (2.0/minus.rows)*minus;
  return res;
}


static cv::Mat maxLoc(const cv::Mat& matx)
{
  // 列向量
  using namespace std;
  cv::Mat maxs(1, matx.cols, CV_32FC1, cv::Scalar(0));
  for (int c_ = 0; c_ < matx.cols; ++c_) {
    for (int r_ = 1; r_ < matx.rows; ++r_) {
      if (matx.at<float>(r_, c_) > maxs.at<float>(0,c_)) {
        maxs.at<float>(0, c_) = matx.at<float>(r_, c_);
      }
    }
  }
  return maxs;

  
  //cv::Mat sorted_index, sorted_matx;
  //cv::sortIdx(matx, sorted_index,
  //  cv::SORT_EVERY_COLUMN + cv::SORT_DESCENDING);
  //cv::sort(matx, sorted_matx,
  //  cv::SORT_EVERY_COLUMN + cv::SORT_DESCENDING);


  //cv::Mat indexes = cv::Mat_<float>(sorted_index);
  //cv::Mat maxloc = indexes.row(0);

  /*cout << "sorted_index:" << endl;
  cout << sorted_index << endl;
  cout << "sorted_matx:" << endl;
  cout << sorted_matx << endl;
  cout << "indexes:" << endl;
  cout << indexes << endl;
  cout << "maxloc:" << endl;
  cout << maxloc << endl;*/


  //return maxloc;
}


#define CHKMat(x) {\
  std::cout<<#x<<":"<<std::endl;\
  std::cout << (x) << std::endl;\
};

// for array
#define CHKMats(x,i){\
  std::cout<<#x<<"["<<i<<"]:"<<std::endl;\
  std::cout<<(x[i])<<std::endl;\
}
