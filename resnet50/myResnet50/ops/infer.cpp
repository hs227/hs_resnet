#include"infer.h"
#include"conv.h"
#include"pooling.h"
#include"bn.h"
#include"fc.h"
#include"softmax.h"











//int main1(void)
//{
//  vector<int> test({1, 3, -1, 4});
//  int test1[] = { -1,1,-2,2 };
//  relu(test);
//  for (int i = 0; i < 4; ++i)
//    printf("%d ", test[i]);
//  printf("\n");
//  relu(test1,4);
//  for (int i = 0; i < 4; ++i)
//    printf("%d ", test1[i]);
//  printf("\n");
//  printf("add:\n");
//  vector<int> test2(test1, test1+sizeof(test1) / sizeof(int));
//  vector<int> test3 = add(test, test2);
//  for (int i = 0; i < 4; ++i)
//    printf("%d ", test3[i]);
//  printf("\n");
//
//
//  return 0;
//}


//void Process(void) 
//{
//  string p_str = "INFER_NET";
//  bool is_down_sample = true;
//  bool is_softmax = true;
//  DataStream<float> input(2, 5, 5, 3,DS_RAND);
//  DataStream<float> output;
//  input.ShowTravel();
//  printf("\n\n\n");
//  output = ComputeConvLayer(input, p_str+"_conv1");
//  output.ShowTravel();
//  output = ComputeBNLayer(output, p_str+"_bn1");
//  output.ShowTravel();
//  output = ComputeReluLayer(output, p_str + "_relu1");
//  output.ShowTravel();
//  output = ComputeConvLayer(input, p_str + "_conv2");
//  output.ShowTravel();
//  output = ComputeBNLayer(output, p_str + "_bn2");
//  output.ShowTravel();
//  output = ComputeReluLayer(output, p_str + "_relu2");
//  output.ShowTravel();
//  output = ComputeConvLayer(input, p_str + "_conv3");
//  output.ShowTravel();
//  output = ComputeBNLayer(output, p_str + "_bn3");
//  output.ShowTravel();
//
//
//  if (is_down_sample) {
//    DataStream<float> conv_out;
//    conv_out= ComputeConvLayer(input, p_str + "_downsample_conv2d");
//    conv_out.ShowTravel();
//    DataStream<float> short_cut_out;
//    short_cut_out = ComputeBNLayer(conv_out, p_str + "_downsample_batchnorm");
//    short_cut_out.ShowTravel();
//    output = ComputeAddLayer(output, short_cut_out, p_str + "_downsample_add");
//    output.ShowTravel();
//  }else {
//    output = ComputeAddLayer(output, input, p_str + "_add");
//    output.ShowTravel();
//  }
//  output = ComputeReluLayer(output, p_str + "_final_relu");
//  output.ShowTravel();
//  output = ComputeFCLayer(output, p_str + "_fully_connected");
//  output.ShowTravel();
//  
//  if(is_softmax){
//    DataStream<double> probs;
//    probs= ComputeSMLayer(output, p_str + "_softmax");
//    probs.ShowTravel();
//  }
//  else {
//    output = ComputeFMLayer(output, p_str + "_findmax");
//    output.ShowTravel();
//  }
//  
//
//  getchar();
//  exit(EXIT_SUCCESS);
//}


//int main(void) 
//{
//  double sums = 0;
//  sums++;
//  int b = 3 / sums;
//  Process();
//  DataStream<float> input(2, 5, 5, 3, DS_RAND);
//  vector<vector<float>> mi= input.datas[0].data[0];
//  printMat2D(mi);
//  unsigned int padding[4] = { 0,0,0,0 };
//  unsigned int stride[2] = { 1,1 };
//  vector<vector<vector<float>>> wMatx = {
//  {
//   {1,0,0,},
//   {0,1,0,},
//   {0,0,1,},
//  },
//  {
//   {1,0,0,},
//   {0,1,0,},
//   {0,0,1,},
//  },
//  {
//   {1,0,0,},
//   {0,1,0,},
//   {0,0,1,},
//  },
//  };
//  //printMat2D(wMatx[0]);
//  //float* mit = vvt2t(mi, &mit);
//  //float* wt = vvt2t(wMatx[0], &wt);
//  //printMat2D(mit, 5, 5);
//  //printMat2D(wt, 3, 3);
//  //unsigned int ow, oh;
//  //conv2D<float>((float*)mit, 5, 5, (float*)wt, 3, 3, padding, stride, &ow, &oh);
//  //vector<vector<float>> picc = conv2D(mi, wMatx[0], padding, stride);
//  //printMat2D(picc);
//
//
//
//  //input.ShowTravel();
//  //printf("\n\n\n");
//  //DataStream<float> output=ComputeConvLayer(input, "conv1");
//  //output.ShowTravel();
//
//  /*vector<vector<int>> c(3, vector<int>(3, 0));
//  vvt2t(c, (int**)&a1);
//  printMat2D(a1, 3, 3);
//  delete[] a1;
//  vector<vector<int>> b;
//  t2vvt<int>((int*)a, b, 3, 3);
//  printMat2D(b);*/
//  //vector<vector<vector<int>>> arr3 = vector<vector<vector<int>>>(
//  //  3, vector<vector<int>>(3, vector<int>(3)));
//
//  return 0;
//}





















