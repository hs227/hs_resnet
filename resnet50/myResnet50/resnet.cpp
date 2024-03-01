#include"resnet.h"
#include"./ops/infer.h"
#include"./ops/conv.h"
#include"./ops/pooling.h"
#include"./ops/bn.h"
#include"./ops/fc.h"
#include"./ops/softmax.h"

#pragma warning(disable:4996)


template<typename T>
static T* LoadDataFromFile(const std::string& file_name, int len,bool is_float) 
{
  T* data = (T*)malloc(len * sizeof(T));
  FILE* fp = fopen(file_name.c_str(), "r");
  // std::cout << "file_name = " << file_name << ", fp = " << fp << std::endl;
  for (int i = 0; i < len; ++i) {
    float x = 0;
    fscanf(fp, "%f", &x);
    data[i] = is_float ? x : (int)x;
  }
  fclose(fp);
  return data;
}


int* LoadConvPara(const std::string& name, int len) 
{
  string file_name = "./src/resnet50/model/resnet50_weight/resnet50_" + name + "_param.txt";
  return LoadDataFromFile<int>(file_name, len, false);
}

template<typename T>
void LoadConvWeight(const std::string& name,
  vector<vector<vector<vector<T>>>>& weights
  ) 
{
  // 能够确保T为float
  //static_assert(std::is_same<T, float>::value, "T must be float");

  std::string file_name = "./src/resnet50/model/resnet50_weight/resnet50_" + name + "_weight.txt";
  
  const unsigned int n = weights.size();
  const unsigned int c = weights[0].size();
  const unsigned int h = weights[0][0].size();
  const unsigned int w = weights[0][0][0].size();

  int len = n * c * h * w;
  float* data = LoadDataFromFile<float>(file_name, len, true);

  for (int n_ = 0; n_ < n; ++n_) {
    for (int c_ = 0; c_ < c; ++c_) {
      for (int h_ = 0; h_ < h; ++h_) {
        for (int w_ = 0; w_ < w; ++w_) {
          weights[n_][c_][h_][w_] = (T)data[n_ * c * h * w + c_ * h * w + h_ * w + w_];
        }
        //printVec(&data[n_ * c * h * w + c_ * h * w + h_ * w], w);
        //printVec(weights[n_][c_][h_]);
      }
    }
  }
  free(data);

}

void LoadBNParam(const std::string& name,int c,
  vector<float>& gamma, vector<float>& beta, vector<float>& mean, vector<float>& var) 
{
  std::string weight_file_name = "./src/resnet50/model/resnet50_weight/resnet50_" + name + "_weight.txt";
  std::string bias_file_name = "./src/resnet50/model/resnet50_weight/resnet50_" + name + "_bias.txt";
  std::string mean_file_name = "./src/resnet50/model/resnet50_weight/resnet50_" + name + "_running_mean.txt";
  std::string var_file_name = "./src/resnet50/model/resnet50_weight/resnet50_" + name + "_running_var.txt";
  float* g = LoadDataFromFile<float>(weight_file_name, c,true);
  float* b = LoadDataFromFile<float>(bias_file_name, c, true);
  float* m = LoadDataFromFile<float>(mean_file_name, c, true);
  float* v = LoadDataFromFile<float>(var_file_name, c, true);
  for (int c_ = 0; c_ < c; ++c_) {
    gamma.push_back(g[c_]);
    beta.push_back(b[c_]);
    mean.push_back(m[c_]);
    var.push_back(v[c_]);
  }
  //printVec(gamma);
  //printVec(beta);
  //printVec(mean);
  //printVec(var);

  free(g);
  free(b);
  free(m);
  free(v);
}

template<typename T>
void LoadFCParam(const std::string& name,
  vector<vector<T>>& weight, vector<T>& bias) 
{
  string weight_file_name ="./src/resnet50/model/resnet50_weight/resnet50_" + name + "_weight.txt";
  string bias_file_name = "./src/resnet50/model/resnet50_weight/resnet50_" + name + "_bias.txt";
  const int wh = weight.size();
  const int ww = weight[0].size();
  const int bw = bias.size();

  float* w = LoadDataFromFile<float>(weight_file_name, wh*ww, true);
  float* b = LoadDataFromFile<float>(bias_file_name, bw, true);
  for (int wh_ = 0; wh_ < wh; ++wh_) {
    for (int ww_ = 0; ww_ < ww; ++ww_) {
      weight[wh_][ww_] = w[wh_ * ww + ww_];
    }
  }
  for (int bw_ = 0; bw_ < bw; ++bw_) {
    bias[bw_] = b[bw_];
  }

  free(w);
  free(b);
  return;
}



template<typename T>
DataStream<T> ComputeConvLayer(DataStream<T>& in_data, const string& layer_name)
{
  // resnet50 中conv不使用bias
  
  // 1.Conv层自身参数
  int* param=LoadConvPara(layer_name, 5);
  // ci, co, kernel, stride, pad
  int p_ci = param[0];
  int p_co = param[1];
  int p_kernel = param[2];
  int p_stride = param[3];
  int p_pad = param[4];
  free(param);

  // 0.运行交互
  printf("In %s computing-- (ci=%d,co=%d,k={%d,%d},s={%d,%d},p={%d，%d})\n", layer_name.c_str(),
    p_ci, p_co, p_kernel,p_kernel,p_stride, p_stride, p_pad, p_pad);
  printf("In NCHW(%d,%d,%d,%d)\n", in_data.n, in_data.channel, in_data.height, in_data.width);


  vector<vector<vector<vector<T>>>> wMatxs(p_co,
    vector<vector<vector<T>>>(p_ci,
      vector<vector<T>>(p_kernel,
        vector<T>(p_kernel))));
  LoadConvWeight<T>(layer_name, wMatxs);
  unsigned int padding[4] = { p_pad,p_pad,p_pad,p_pad };
  unsigned int stride[2] = { p_stride,p_stride };
  //T bias = 0;


  // 2.NHWC参数计算
  const unsigned int in = in_data.n;
  const unsigned int ih = in_data.height + padding[0] + padding[1];
  const unsigned int iw = in_data.width + padding[2] + padding[3];
  const unsigned int ic = in_data.channel;
  //printf("in(%d,%d,%d,%d)\n", in, ih, iw, ic);

  const unsigned int wn = wMatxs.size();
  const unsigned int wh = wMatxs[0][0].size();
  const unsigned int ww = wMatxs[0][0][0].size();
  const unsigned int wc = wMatxs[0].size();
  //printf("w(%d,%d,%d,%d)\n", wn, wh, ww, wc);

  const unsigned int stride_height = stride[0];
  const unsigned int stride_width = stride[1];

  const unsigned int on = in;
  const unsigned int oh = (ih - wh) / stride_height + 1;
  const unsigned int ow = (iw - ww) / stride_width + 1;
  const unsigned int oc = wn;
  //printf("out(%d,%d,%d,%d)\n", on, oh, ow, oc);

  DataStream<T> out_data(on, ow, oh, oc);
  //vector<vector<T>> biasMatx(oh, vector<T>(ow, bias));
  // 3.Conv

  for (unsigned int on_ = 0; on_ < on; ++on_) {
    for (unsigned int oc_ = 0; oc_ < oc; ++oc_) {
      //vector<vector<vector<T>>>& wMatx = wMatxs[oc_];
      vector<vector<T>> pic(oh, vector<T>(ow, 0));
      for (unsigned int ic_ = 0; ic_ < ic; ++ic_) {
        //vector<vector<T>>& wMatxc = wMatx[ic_];
        //vector<vector<T>> picc = conv2D(in_data.datas[on_].data[ic_], wMatxc, padding, stride);
        vector<vector<T>> picc = conv2D(in_data.datas[on_].data[ic_], wMatxs[oc_][ic_], padding, stride);
        vvtplus(pic, picc, pic);
      }
      //vvtplus(pic, biasMatx, out_data.datas[on_].data[oc_]);
    }
  }

  printf("Out NCHW(%d,%d,%d,%d)\n", out_data.n, out_data.channel, out_data.height, out_data.width);
  printf("Out %s \n", layer_name.c_str());
  return out_data;
}

template<typename T>
DataStream<T> ComputeBNLayer(DataStream<T>& in_data, const string& layer_name)
{
  // 未实现momentum，affine，track_running_stats
  // 1.BN层自身参数
  vector<T> gamma;
  vector<T> beta;
  vector<T> mean;
  vector<T> var;
  LoadBNParam(layer_name, in_data.channel, gamma, beta, mean, var);

  // 0.运行交互
  printf("In %s computing--- (c=%d)\n", layer_name.c_str(), in_data.channel);
  printf("In NCHW(%d,%d,%d,%d)\n", in_data.n, in_data.channel, in_data.height, in_data.width);

  // 2.NHWC参数计算
  const unsigned int n = in_data.n;
  const unsigned int h = in_data.height;
  const unsigned int w = in_data.width;
  const unsigned int c = in_data.channel;
  // 3.运行
  DataStream<T> out_data(n, w, h, c);
  for (unsigned int n_ = 0; n_ < n; ++n_) {
    for (unsigned int c_ = 0; c_ < c; ++c_) {
      out_data.getpicc(n_, c_) = batch_norm(in_data.datas[n_].data[c_], gamma[c_], beta[c_],mean[c_], var[c_]);
    }
  }

  printf("Out NCHW(%d,%d,%d,%d)\n", out_data.n, out_data.channel, out_data.height, out_data.width);
  printf("Out %s \n", layer_name.c_str());
  return out_data;
}

template<typename T>
DataStream<T> ComputeReluLayer(DataStream<T>& in_data, const string& layer_name)
{
  // 0.运行交互
  std::cout << "In " << layer_name << " computing---" << std::endl;
  printf("In NCHW(%d,%d,%d,%d)\n", in_data.n, in_data.channel, in_data.height, in_data.width);
  // 1.Relu层自身参数
  // NULL
  // 2.NHWC参数计算
  const unsigned int n = in_data.n;
  const unsigned int h = in_data.height;
  const unsigned int w = in_data.width;
  const unsigned int c = in_data.channel;
  // 3.运行
  DataStream<T> out_data(n, w, h, c);
  for (unsigned int n_ = 0; n_ < n; ++n_) {
    for (unsigned int c_ = 0; c_ < c; ++c_) {
      for (unsigned int h_ = 0; h_ < h; ++h_) {
        out_data.datas[n_].data[c_][h_] = relu(in_data.datas[n_].data[c_][h_]);
      }
    }
  }
  printf("Out NCHW(%d,%d,%d,%d)\n", out_data.n, out_data.channel, out_data.height, out_data.width);
  printf("Out %s \n", layer_name.c_str());
  return out_data;
}

template<typename T>
DataStream<T> ComputeMaxPoolLayer(DataStream<T>& in_data, const string& layer_name)
{
  //未实现dilation

  // 1.MaxPool层自身参数
  int kernel = 3;
  int stride = 2;
  int pad = 1;
  unsigned int pads[4] = { pad, pad, pad, pad };
  unsigned int strides[2] = { stride,stride};

  // 0.运行交互
//std::cout << "In " << layer_name << " computing---" << std::endl;
  printf("In %s computing-- (k={%d,%d},s={%d,%d},p={%d，%d})\n", layer_name.c_str(),
    kernel, kernel, stride, stride, pad, pad);
  printf("In NCHW(%d,%d,%d,%d)\n", in_data.n, in_data.channel, in_data.height, in_data.width);

  
  // 2.NHWC参数计算
  const unsigned int in = in_data.n;
  const unsigned int ih = in_data.height;
  const unsigned int iw = in_data.width;
  const unsigned int ic = in_data.channel;

  const unsigned int on = in;
  const unsigned int oh = (ih + 2 * pad - kernel) / stride + 1;
  const unsigned int ow = (iw + 2 * pad - kernel) / stride + 1;
  const unsigned int oc = ic;

  // 3.运行
  DataStream<T> out_data(on, ow, oh, oc);
  for (unsigned int n_ = 0; n_ < on; ++n_) {
    for (unsigned int c_ = 0; c_ < oc; ++c_) {
      out_data.datas[n_].data[c_] = pool2D(
        in_data.datas[n_].data[c_], kernel, kernel, pads, strides, POOLTYPE::MAXPOOLING);
    }
  }

  printf("Out NCHW(%d,%d,%d,%d)\n", out_data.n, out_data.channel, out_data.height, out_data.width);
  printf("Out %s \n", layer_name.c_str());
  return out_data;
}

template<typename T>
DataStream<T> ComputeAddLayer(DataStream<T>& in_data1, DataStream<T>& in_data2, const string& layer_name)
{
  // 0.运行交互
  printf("In %s computing---\n", layer_name.c_str());
  // 1.Add层自身参数
  // NULL
  // 2.NHWC参数计算
  const unsigned int n = in_data1.n;
  const unsigned int h = in_data1.height;
  const unsigned int w = in_data1.width;
  const unsigned int c = in_data1.channel;
  // 3.运行
  DataStream<T> out_data(n, w, h, c);
  for (unsigned int n_ = 0; n_ < n; ++n_) {
    for (unsigned int c_ = 0; c_ < c; ++c_) {
      for (unsigned int h_ = 0; h_ < h; ++h_) {
        out_data.datas[n_].data[c_][h_] = add(in_data1.datas[n_].data[c_][h_], in_data2.datas[n_].data[c_][h_]);
      }
    }
  }
  printf("Out %s \n", layer_name.c_str());
  return out_data;
}


template<typename T>
DataStream<T> ComputeBottleNeck(DataStream<T>& input, const string& layer_name, bool down_sample) 
{
  DataStream<T> output;
  printf("In %s computing---\n", layer_name.c_str());

  output = ComputeConvLayer(input, layer_name + "_conv1");
  output = ComputeBNLayer(output, layer_name + "_bn1");
  output = ComputeReluLayer(output, layer_name + "_relu1");
  output = ComputeConvLayer(output, layer_name + "_conv2");
  output = ComputeBNLayer(output, layer_name + "_bn2");
  output = ComputeReluLayer(output, layer_name + "_relu2");
  output = ComputeConvLayer(output, layer_name + "_conv3");
  output = ComputeBNLayer(output, layer_name + "_bn3");


  if (down_sample) {
    DataStream<T> conv_out;
    conv_out = ComputeConvLayer(input, layer_name + "_downsample_conv2d");
    DataStream<T> short_cut_out;
    short_cut_out = ComputeBNLayer(conv_out, layer_name + "_downsample_batchnorm");
    output = ComputeAddLayer(output, short_cut_out, layer_name + "_downsample_add");
  }
  else {
    output = ComputeAddLayer(output, input, layer_name + "_add");
  }
  printf("Out NCHW(%d,%d,%d,%d)\n", output.n, output.channel, output.height, output.width);
  printf("Out %s \n\n\n", layer_name.c_str());
  return output;
}


template<typename T>
DataStream<T> ComputeAvePoolLayer(DataStream<T>& in_data, const string& layer_name)
{

  // 1.AvePool层自身参数
  unsigned int output[2] = { 1,1 };
  unsigned int pads[4] = { 0,0,0,0 };
  unsigned int strides[2] = { 1,1 };

  // 0.运行交互
//std::cout << "In " << layer_name << " computing---" << std::endl;
  printf("In %s computing-- (output={1,1})\n", layer_name.c_str());
  printf("In NCHW(%d,%d,%d,%d)\n", in_data.n, in_data.channel, in_data.height, in_data.width);


  // 2.NHWC参数计算
  const unsigned int in = in_data.n;
  const unsigned int ih = in_data.height;
  const unsigned int iw = in_data.width;
  const unsigned int ic = in_data.channel;

  const unsigned int on = in;
  const unsigned int oh = output[0];
  const unsigned int ow = output[1];
  const unsigned int oc = ic;

  // 3.运行
  DataStream<T> out_data(on, ow, oh, oc);
  for (unsigned int n_ = 0; n_ < on; ++n_) {
    for (unsigned int c_ = 0; c_ < oc; ++c_) {
      out_data.datas[n_].data[c_] = pool2D(
        in_data.datas[n_].data[c_], iw/ow, ih/oh, pads, strides, POOLTYPE::AVEPOOLING);
    }
  }

  printf("Out NCHW(%d,%d,%d,%d)\n", out_data.n, out_data.channel, out_data.height, out_data.width);
  printf("Out %s \n", layer_name.c_str());
  return out_data;
}

template<typename T>
vector<vector<T>> flatten(DataStream<T>& input) 
{
  const int n = input.n;
  const int c = input.channel;
  vector<vector<T>> flats(n, vector<T>(c));
  for (int n_ = 0; n_ < n; ++n_) {
    for (int c_ = 0; c_ < c; ++c_) {
      flats[n_][c_] = input.datas[n_].data[c_][0][0];
    }
  }
  return flats;
}

template<typename T>
DataStream<T> ComputeFCLayer(DataStream<T>& in_data, const string& layer_name)
{
  // 1.FC层自身参数
  int in_features = in_data.channel;
  int out_features = 1000;
  vector<vector<T>> weight(out_features, vector<T>(in_features));// 1000*2048
  vector<T> bias(out_features);// 1*1000
  LoadFCParam(layer_name,weight, bias);

  // 2.NHWC参数计算
  const unsigned int on = in_data.n;
  //const unsigned int oh = 1;
  const unsigned int ow = out_features;
  //const unsigned int oc = 1;
  
  // 0.运行交互
  printf("In %s computing--(in=%d,out=%d)\n", layer_name.c_str(), in_features, out_features);
  printf("In NCHW(%d,%d,%d,%d)\n", in_data.n, in_data.channel, in_data.height, in_data.width);
  
  
  // 3.运行 (行向量)
  DataStream<T> out_data(on, ow, 1, 1);
  vector<vector<T>> flats = flatten(in_data);
  
  for (unsigned int n_ = 0; n_ < on; ++n_) {
    out_data.datas[n_].data[0][0] = fc2D(flats[n_], weight, bias);
  }

  printf("Out NCHW(%d,%d,%d,%d)\n", out_data.n, out_data.channel, out_data.height, out_data.width);
  printf("Out %s \n", layer_name.c_str());
  return out_data;
}

template<typename T>
DataStream<T> ComputeFMLayer(DataStream<T>& in_data, const string& layer_name)
{
  // 0.运行交互
  std::cout << "In " << layer_name << " computing---" << std::endl;
  // 1.findMax层自身参数
  // 2.NHWC参数计算
  const unsigned int n = in_data.n;
  const unsigned int h = in_data.height;
  const unsigned int w = in_data.width;
  const unsigned int c = in_data.channel;
  // 3.运行 
  DataStream<T> out_data(n, 1, h, c);
  for (unsigned int n_ = 0; n_ < n; ++n_) {
    for (unsigned int c_ = 0; c_ < c; ++c_) {
      for (unsigned int h_ = 0; h_ < h; ++h_) {
        vector<T> oneRow = in_data.datas[n_].data[c_][h_];
        out_data.datas[n_].data[c_][h_][0] = in_data.datas[n_].data[c_][h_][findMax(oneRow)];
      }
    }
  }

  return out_data;
}


template<typename T>
DataStream<double> ComputeSMLayer(DataStream<T>& in_data, const string& layer_name)
{
  // 0.运行交互
  std::cout << "In " << layer_name << " computing---" << std::endl;
  // 1.SoftMax层自身参数
  // 2.NHWC参数计算
  const unsigned int n = in_data.n;
  const unsigned int h = in_data.height;
  const unsigned int w = in_data.width;
  const unsigned int c = in_data.channel;
  // 3.运行 
  DataStream<double> out_data(n, 1, h, c);
  for (unsigned int n_ = 0; n_ < n; ++n_) {
    for (unsigned int c_ = 0; c_ < c; ++c_) {
      for (unsigned int h_ = 0; h_ < h; ++h_) {
        vector<T> oneRow = in_data.datas[n_].data[c_][h_];
        printVec(oneRow);
        vector<double>probs = softmax(oneRow);
        printVec(probs);
        out_data.datas[n_].data[c_][h_][0] = probs[findMax(probs)];
      }
    }
  }

  return out_data;
}




DataStream<float> Resnet::run(DataStream<float>& input)
{
  DataStream<float> output;

  //(conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  output = ComputeConvLayer(input, "conv1");
  //(bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  output = ComputeBNLayer(output, "bn1");
  output = ComputeReluLayer(output, "relu1");
  output = ComputeMaxPoolLayer(output, "maxpool");
  // layer1
  output = ComputeBottleNeck(output, "layer1_bottleneck0", true);
  output = ComputeBottleNeck(output, "layer1_bottleneck1", false);
  output = ComputeBottleNeck(output, "layer1_bottleneck2", false);
  // layer2
  output = ComputeBottleNeck(output, "layer2_bottleneck0", true);
  output = ComputeBottleNeck(output, "layer2_bottleneck1", false);
  output = ComputeBottleNeck(output, "layer2_bottleneck2", false);
  output = ComputeBottleNeck(output, "layer2_bottleneck3", false);
  // layer3
  output = ComputeBottleNeck(output, "layer3_bottleneck0", true);
  output = ComputeBottleNeck(output, "layer3_bottleneck1", false);
  output = ComputeBottleNeck(output, "layer3_bottleneck2", false);
  output = ComputeBottleNeck(output, "layer3_bottleneck3", false);
  output = ComputeBottleNeck(output, "layer3_bottleneck4", false);
  output = ComputeBottleNeck(output, "layer3_bottleneck5", false);
  // layer4
  output = ComputeBottleNeck(output, "layer4_bottleneck0", true);
  output = ComputeBottleNeck(output, "layer4_bottleneck1", false);
  output = ComputeBottleNeck(output, "layer4_bottleneck2", false);
  // avg pool
  output = ComputeAvePoolLayer(output, "avgpool");
  // Linear
  output = ComputeFCLayer(output, "fc");










  return output;
}

#include"../label.h"

/* 打印运行结果 */
void showRes(vector<float>& res) {
  std::vector<std::pair<float, int>> sort_pairs;
  for (int i = 0; i < res.size(); ++i) {
    sort_pairs.emplace_back(res[i], i);
  }

  std::sort(sort_pairs.begin(), sort_pairs.end(),
    [](const pair<float, int>& a, const pair<float, int>& b) {
      return a.first > b.first;
    });

  auto labels = load_imagenet_labels();
  const int topk = 5;
  for (int i = 0; i < topk; ++i) {
    std::cout << "top " << (i + 1) << " " << sort_pairs[i].first << " -> Index=["
      << sort_pairs[i].second << "]"
      << ", Label=[" << labels[sort_pairs[i].second] << "]" << std::endl;
  }
  std::cout << "\n" << std::endl;

}

void Resnet::printRes(DataStream<float>& output)
{
  const unsigned int n = output.n;

  for (int n_ = 0; n_ < n; ++n_) {
    showRes(output.datas[n_].data[0][0]);
  }
  return;
}




