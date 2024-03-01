#pragma once
#include<iostream>
#include<opencv2\opencv.hpp>
#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<vector>


enum InitType {
  DefInit=0,
  RandInit,
  SeqInit,
};
enum ActivType {
  SigAT=0,
  TanhAT,
  ReluAT,
  NullAT,
};



class NEUNet 
{
public:
  std::vector<int> layer_neuron_num;
  enum ActivType activation_type = ReluAT;
  float learning_rate = 2e-2;
  float fine_tune_factor = 1.01;

  
  NEUNet() {}
  ~NEUNet() {}
  NEUNet(const std::vector<int>& neuron_nums)
  {
    if (neuron_nums.size() < 2)
      assert(!"Net create failed(Bad Size)");
    initNet(neuron_nums);
  }

  void showInfo(void);


  void initWData(float l,float r,enum InitType type);
  void initBData(float l,float r, enum InitType type);


  cv::Mat predict(const cv::Mat& input);
  cv::Mat predictOne(const cv::Mat& input);
  

  void trainL(const cv::Mat& input, const cv::Mat& true_out_, const float loss_threshold);
  void test(const cv::Mat& input, const cv::Mat& true_out_);

  void forward(void);
  void backward(void);

  
  void saveMod(const std::string& filename);
  void loadMod(const std::string& filename);

private:
  std::vector<cv::Mat> neurons;
  std::vector<cv::Mat> weights;
  std::vector<cv::Mat> biases;

  // for the backward
  cv::Mat pred_out;
  cv::Mat true_out;

  void initNet(const std::vector<int>& neuron_nums);



  cv::Mat activ(cv::Mat& matx);
  cv::Mat deactiv(cv::Mat& matx);


};





