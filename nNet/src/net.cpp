#include"../include/net.h"
#include"../include/functions.h"

using namespace std;

void NEUNet::showInfo(void) 
{
  cout << "Net Info:" << endl << endl;
  // LAYERS
  std::cout << "Layers"<<
    "("<<layer_neuron_num.size()<<"):"<<endl;
  cout << "{";
  for (int i = 0; i < layer_neuron_num.size(); ++i) {
    cout << ", " << layer_neuron_num[i];
  }
  cout << "}" << endl;


  // NEURONS
  std::cout << "Neurons:" << std::endl;
  for (int i = 0; i < neurons.size(); ++i) {
    cout << "n" << i << ":" << endl;
    cout << neurons[i] << endl;
  }

  // WEIGHTS
  std::cout << "Weights:" << std::endl;
  for (int i = 0; i < weights.size(); ++i) {
    std::cout << "w" << i << ":" << endl;
    cout << weights[i] << endl;
  }

  // BIASES
  std::cout << "Biases:" << std::endl;
  for (int i = 0; i < biases.size(); ++i) {
    std::cout << "b" << i << ":" << endl;
    cout << biases[i] << endl;
  }


  //// pred_out
  //std::cout << "pred_out:" << std::endl;
  //cout << pred_out << std::endl;

  //// true_out
  //std::cout << "true_out:" << std::endl;
  //cout << true_out << endl;

  //// loss
  //std::cout << "loss:" << std::endl;
  //cout << loss << endl;

}

void NEUNet::initWData(float l, float r, enum InitType type)
{
  const int w_len = weights.size();

  switch (type) {
  case InitType::DefInit:
    for (int i = 0; i < w_len; ++i) {
      weights[i].setTo(cv::Scalar(l));
    }
    break;
  case InitType::RandInit:
    for (int i = 0; i < w_len; ++i) {
      cv::randu(weights[i], l, r);
    }
    break;
  case InitType::SeqInit:
    for (int i = 0; i < w_len; ++i) {
      const int rows = weights[i].rows;
      const int cols = weights[i].cols;
      for (int r_ = 0; r_ < rows; ++r_) {
        for (int c_ = 0; c_ < cols; ++c_) {
          weights[i].at<float>(r_, c_) = r_ * 100 + c_;
        }
      }
    }
    break;
  default:
    break;
  }

}

void NEUNet::initBData(float l, float r, enum InitType type)
{
  const int b_len = biases.size();

  switch (type) {
  case InitType::DefInit:
    for (int i = 0; i < b_len; ++i) {
      biases[i].setTo(cv::Scalar(l));
    }
    break;
  case InitType::RandInit:
    for (int i = 0; i < b_len; ++i) {
      cv::randu(biases[i], l, r);
    }
    break;
  case InitType::SeqInit:
    for (int i = 0; i < b_len; ++i) {
      const int rows = biases[i].rows;
      const int cols = biases[i].cols;
      for (int r_ = 0; r_ < rows; ++r_) {
        for (int c_ = 0; c_ < cols; ++c_) {
          biases[i].at<float>(r_, c_) = r_ * 100 + c_;
        }
      }
    }
    break;
  default:
    break;
  }

}


cv::Mat NEUNet::predict(const cv::Mat& input)
{
  // 列向量组
  cv::Mat cols = cv::Mat::zeros(1, 0, CV_32FC1);
  for (int i = 0; i < input.cols; ++i) 
  {
    cv::Mat onecol=predictOne(input.col(i));
    //std::cout << i << std::endl;
    //std::cout << "onecol:" << i << std::endl;
    //std::cout << onecol << std::endl;
    cv::hconcat(cols, onecol,cols);
  }
  //std::cout << "cols:" << std::endl;
  //std::cout << cols << std::endl;
 

  return cols;
}

cv::Mat NEUNet::predictOne(const cv::Mat& input)
{
  // 列向量
  if (input.cols != 1)
    return cv::Mat();
  // 列数不匹配
  if (input.rows != neurons[0].rows)
    return cv::Mat();
  
  neurons[0] = input;
  forward();
  cv::Mat pred_res = neurons[neurons.size() - 1];

  cv::Mat pred_loc = maxLoc(pred_res);
  return pred_loc;
}


void NEUNet::trainL(const cv::Mat& input, const cv::Mat& true_out_, const float loss_threshold) 
{
  const int trainLen = input.cols;
  assert(input.cols == true_out_.cols&&"cols wrong");

  const int maxdepth = 1000;
  const int output_interval = 1;
  const int blast_interval = 100;
  
  float loss_sum = loss_threshold + 1;

  int depth = 0;
  
  while (loss_sum > loss_threshold) {
    loss_sum = 0;

    for (int i = 0; i < trainLen; ++i) {
      neurons[0] = input.col(i);
      true_out = true_out_.col(i);
      showInfo();
      forward();
      showInfo();
      pred_out = neurons[neurons.size() - 1];
      cout << pred_out << endl;
      cout << true_out << endl;
      float loss = mseLoss(pred_out, true_out).at<float>(0,0);
      loss_sum += loss;

      backward();

    }
    depth++;
    if ((depth % output_interval == 0)) {
      printf("depth=%d\n", depth);
      printf("loss_sum=%f\n", loss_sum);
      //showInfo();
      
    }
    if ((depth % blast_interval == 0) ) {
      learning_rate *= fine_tune_factor;
    }


    if (depth >= maxdepth) {
      break;
    }
  }
  printf("Train finished\n");
  printf("depth=%d\n", depth);
  printf("loss_sum=%f\n", loss_sum);
  
}

void NEUNet::test(const cv::Mat& input, const cv::Mat& true_out_)
{
  const int trainLen = input.cols;
  assert(input.cols == true_out_.cols && "cols wrong");
  int p_right_num = 0;
  cv::Mat predict_res = predict(input);
  cv::Mat true_res = maxLoc(true_out_);

  for (int i = 0; i < trainLen; ++i) {
    //printf("Sample%d=p/t(%f,%f)\n", i, predict_res.at<float>(0, i), true_res.at<float>(0, i));
    if (predict_res.at<float>(0, i) == true_res.at<float>(0, i)) {
      p_right_num++;
      printf("Sample%d=p/t(%f,%f)\n", i, predict_res.at<float>(0, i), true_res.at<float>(0, i));

    }
  }

  float accuracy = ((float)p_right_num) / ((float)trainLen);
  printf("Test finished.\n");
  printf("Sample num   =%d\n", trainLen);
  printf("Right predict=%d\n", p_right_num);
  printf("Accuracy     =%f\n", accuracy);
  
}

void NEUNet::initNet(const vector<int>& neuron_nums)
{
  layer_neuron_num = neuron_nums;

  // Init neurons
  neurons.resize(neuron_nums.size());
  const int len_neurons = neurons.size();
  for (int i = 0; i < len_neurons; i++) {
    neurons[i].create(neuron_nums[i], 1, CV_32FC1);
  }
  

  // Init weights/bias
  weights.resize(neuron_nums.size() - 1);
  biases.resize(neuron_nums.size() - 1);
  for (int i = 0; i < len_neurons - 1; ++i) {
    weights[i].create(neuron_nums[i+1], neuron_nums[i], CV_32FC1);
    weights[i].setTo(cv::Scalar(0.5));// 全部初始化为0.5
    biases[i].create(neuron_nums[i+1], 1, CV_32FC1);
    biases[i].setTo(cv::Scalar(0));// 全部初始化为0
  }



}

void NEUNet::forward(void) 
{
  const int layers = neurons.size() - 1;
  // Y=W*X+B
  // Z=activ(Y)
  for (int i = 0; i < layers; ++i) {
    neurons[i + 1] = weights[i] * neurons[i] + biases[i];
    neurons[i + 1] = activ(neurons[i+1]);
  }
}

void NEUNet::backward(void) 
{
  // calc prime
  cv::Mat loss_prime=mseLoss_prime(pred_out, true_out);
  CHKMat(loss_prime);
  cv::Mat predout_prime = deactiv(loss_prime);
  CHKMat(predout_prime);
  

  vector<cv::Mat> neuron_prime(neurons.size());
  for (int i = 0; i < neurons.size(); ++i) {
    neuron_prime[i] = neurons[i].clone();
  }
  vector<cv::Mat> weight_prime(weights.size());
  vector<cv::Mat> bias_prime(biases.size());
  for (int i = 0; i < weights.size(); ++i) {
    weight_prime[i] = weights[i].clone();
    bias_prime[i] = biases[i].clone();
  }

  neuron_prime[neuron_prime.size() - 1] = predout_prime;
  
  for (int i = neuron_prime.size() - 2; i >= 0; i--) {
    neuron_prime[i] = weights[i].t()*neuron_prime[i+1];
    CHKMats(neuron_prime, i);
    neuron_prime[i] = deactiv(neuron_prime[i]);
    CHKMats(neuron_prime, i);
    bias_prime[i] = neuron_prime[i + 1];
    CHKMats(bias_prime, i);
    //for (int r_ = 0; r_ < weight_prime[i].rows; ++r_) {
    //  for (int c_ = 0; c_ < weight_prime[i].cols; ++c_) {
    //    weight_prime[i].at<float>(r_, c_) =
    //      neuron_prime[i + 1].at<float>(r_, 0) *
    //      neurons[i].at<float>(c_, 0);
    //  }
    //}
    weight_prime[i] = neuron_prime[i + 1] * neurons[i].t();
    CHKMats(weight_prime, i);
    
    //cout << "i" << i << ":" << endl;
    //CHKMats(neuron_prime,i);
    //CHKMats(bias_prime,i);
    //CHKMats(weight_prime,i);
  }
  // prime end

  // update weight and bias
  //cout << "Update W&B:" << endl;
  for (int i = neurons.size() - 2; i >= 0; --i) {
    weights[i] = weights[i] - learning_rate * weight_prime[i];
    biases[i] = biases[i] - learning_rate * bias_prime[i];
    //cout << "i" << i << ":" << endl;
    //CHKMats(weights,i);
    //CHKMats(biases,i);
  }
  // update end


}



static map<string, enum ActivType> atRWstr = {
    {"SigAT", SigAT}, { "TanhAT",TanhAT }, { "ReluAT",ReluAT }, {"NullAT",NullAT }
};
static inline const std::string activTypeRW(enum ActivType type)
{
  for (auto& tmp : atRWstr) {
    if (tmp.second == type)
      return tmp.first;
  }
  return string("");
}
static inline const enum ActivType activTypeRW(const string& name )
{
  return atRWstr[name];
}

void NEUNet::saveMod(const std::string& filename)
{
  cv::FileStorage model(filename, cv::FileStorage::WRITE);
  model << "layer_neuron_num" << layer_neuron_num;
  model << "activation_type" << activTypeRW(activation_type);
  model << "learning_rate" << learning_rate;
  
  // weights
  for (int i = 0; i < weights.size(); ++i) {
    std::string weight_name = "weight_" + std::to_string(i);
    model << weight_name << weights[i];
  }
 // bias
  for (int i = 0; i < biases.size(); ++i) {
    std::string bias_name = "bias_" + std::to_string(i);
    model << bias_name << biases[i];
  }

  model.release();
}
void NEUNet::loadMod(const std::string& filename)
{
  cv::FileStorage fs;
  fs.open(filename, cv::FileStorage::READ);

  fs["layer_neuron_num"] >> layer_neuron_num;
  {
    string name;
    fs["activation_type"] >> name;
    activation_type = activTypeRW(name);
  }
  fs["learning_rate"] >> learning_rate;
  initNet(layer_neuron_num);

  // weights
  for (int i = 0; i < weights.size(); ++i) {
    std::string weight_name = "weight_" + std::to_string(i);
    fs[weight_name] >> weights[i];
  }
  // bias
  for (int i = 0; i < biases.size(); ++i) {
    std::string bias_name = "bias_" + std::to_string(i);
    fs[bias_name] >> biases[i];
  }
  fs.release();
}


cv::Mat NEUNet::activ(cv::Mat& matx)
{
  cv::Mat res;
  switch (activation_type) {
  case SigAT:
    res = sigmoid(matx);
    break;
  case TanhAT:
    res = tanh(matx);
    break;
  case ReluAT:
    res = relu(matx);
    break;
  case NullAT:
    res = matx;
  default:
    break;
  }
  return res;
}

cv::Mat NEUNet::deactiv(cv::Mat& matx)
{
  cv::Mat res;
  switch (activation_type) {
  case SigAT:
    res = sigmoid_prime(matx);
    break;
  case TanhAT:
    res = tanh_prime(matx);
    break;
  case ReluAT:
    res = relu_prime(matx);
    break;
  case NullAT:
    res = cv::Mat::ones(matx.size(),CV_32FC1);
  default:
    break;
  }
  return res;
}

