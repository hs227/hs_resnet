// BUG here
#include<opencv2/opencv.hpp>
#include<opencv2/ml.hpp>
#include<iostream>

using namespace std;
using namespace cv;

static int e_w = 28;
static int e_h = 28;

void drawSample(Mat& input_, int idx, Mat& board, int px, int py)
{
  // MNIST 每一个样本都是28*28的图像

  Mat digit = board(Rect(px, py, 28, 28));
  Mat sample = input_.col(idx);
  printf("sample[%d]=(%d,%d):drawing\n", idx, sample.rows, sample.cols);
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      float num = sample.at<float>(i * 28 + j);
      digit.at<float>(i, j) = num;
      printf("%d ", (int)num);
    }
    printf("\n");
  }


}

void drawSamples(Mat& input_, int left, int right, int col_strider)
{
  if (left == right)
    return;
  int d_nums = right - left;
  int b_rows = (d_nums + col_strider - 1) / col_strider;
  int b_cols = b_rows == 1 ? d_nums : d_nums - (b_rows - 1) * col_strider;
  int b_width = max(b_cols, col_strider) * e_w;
  int b_height = b_rows * e_h;
  Mat board(b_height, b_width, CV_32FC1);

  for (int i = 0; i < b_rows - 1; ++i) {
    for (int j = 0; j < col_strider; ++j) {
      drawSample(input_, left + i * col_strider + j, board, j * e_w, i * e_h);
    }
  }
  for (int i = 0; i < b_cols; ++i)
  {
    drawSample(input_, left + (b_rows - 1) * col_strider + i, board, i * e_w, (b_rows - 1) * e_h);
  }
  imshow("Samples", board);
  waitKey();
}

int csv2xml()
//int main()
{
  // 读取CSV文件
  Ptr<ml::TrainData>data = NULL;
  data = ml::TrainData::loadFromCSV("mnist_train.csv", 1);
  if (data == NULL) {
    return 0;
  }
  // 获取数据和标签
  Mat samples_ = data->getSamples();//0~size()-2
  Mat responses_ = data->getResponses();//size()-1

  Mat label_ = samples_.col(0);
  Mat input_;
  cv::hconcat(samples_.colRange(1, samples_.cols), responses_, input_);
  input_ = input_.t();
  // label=0
  // input=1~size()-1

  
  // 初始化目标矩阵
  Mat target_(10, input_.cols, CV_32F, Scalar::all(0.));

  // 遍历标签，将对应位置的值设为1
  for (int i = 0; i < label_.rows; ++i)
  {
    float label_num = label_.at<float>(i, 0);
    target_.at<float>(label_num, i) = label_num;
  }

  // 归一化输入数据
  Mat input_normalized;
  input_.convertTo(input_normalized, CV_32F, 1.0 / 255);

  // 将数据写入XML文件 
  //FileStorage fs("./data/input_label_0-9.xml", FileStorage::WRITE);
  //fs << "input" << input_normalized;
  //fs << "target" << target_;
  //fs.release();


  // 提取前1000个样本
  //Mat input_1000 = input_normalized(Rect(0, 0, 1000, input_normalized.rows));
  //Mat target_1000 = target_(Rect(0, 0, 1000, target_.rows));
  Mat input_1000 = input_normalized.colRange(0, 1000);
  Mat target_1000 = target_.colRange(0, 1000);


  // 将前10000个样本写入另一个XML文件
  FileStorage fs2("./data/input_label_0-9_1000.xml", FileStorage::WRITE);
  fs2 << "input" << input_1000;
  fs2 << "target" << target_1000;
  fs2.release();

  return 0;
}








