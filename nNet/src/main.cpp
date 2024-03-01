#include"../include/net.h"
#include"../include/functions.h"

using namespace std;

#define CHKMat(x) {\
  std::cout<<#x<<":"<<std::endl;\
  std::cout << (x) << std::endl;\
};


static void readSamples(const std::string& filename, cv::Mat& input, cv::Mat& target, int startpos, int sample_num) 
{
  cv::FileStorage fs;
  fs.open("./data/input_label_0-9_1000.xml", cv::FileStorage::READ);
  cv::Mat input_, target_;
  fs["input"] >> input_;
  fs["target"] >> target_;
  fs.release();

  input = input_(cv::Rect(startpos, 0, sample_num, input_.rows));
  target = target_(cv::Rect(startpos, 0, sample_num, target_.rows));

}



int main1(void)
{
  NEUNet net3({ 3,2,2 });
  //net3.initWData(0, 0, SeqInit);
  //net3.initBData(0, 0, SeqInit);
  
  cv::Mat input;
  input.create(3, 1, CV_32FC1);
  input.at<float>(0, 0) = 4;
  input.at<float>(1, 0) = 2;
  input.at<float>(2, 0) = 8;

  cv::Mat true_out;
  true_out.create(2, 1, CV_32FC1);
  true_out.at<float>(0, 0) = 1;
  true_out.at<float>(1, 0) = 0;
  

  net3.trainL(input, true_out, 1e-12);
  //net3.predictOne(input, true_out);
  //net3.backward();

  net3.showInfo();
  net3.saveMod("./models/model1.xml");
  
  NEUNet net33;
  net33.loadMod("./models/model1.xml");
  net33.trainL(input, true_out, 1e-12);

  return 0;
}

int main(void) 
{
  NEUNet net3({ 3,2,2 });
  //net3.showInfo();
  
  cv::Mat input;
  input.create(3, 1, CV_32FC1);
  input.at<float>(0, 0) = 4;
  input.at<float>(1, 0) = 2;
  input.at<float>(2, 0) = 8;
  cv::Mat target;
  target.create(2, 1, CV_32FC1);
  target.at<float>(0, 0) = 1;
  target.at<float>(1, 0) = 0;


  net3.trainL(input,target,1e-12);
  
  
  //std::cout << input << std::endl;
  //std::cout << true_out << std::endl;

  //cv::Mat mse = mseLoss(input, true_out);
  //std::cout << mse << std::endl;
 

  return 0;
}

int main123(void) 
{
  cv::Mat input, target;
  readSamples("./data/input_label_0-9_1000.xml",input,target,0,200);
  cv::Mat testInput, testTarget;
  readSamples("./data/input_label_0-9_1000.xml", testInput,testTarget,201,100);
  //cout << input << endl;
  //cout << target << endl;
  //cout << testInput << endl;
  //cout << testTarget << endl;


  NEUNet net3({ 784,100,100,10 });
  net3.initWData(0.01, 0, InitType::DefInit);
  net3.trainL(input, target, 100);
  net3.saveMod("./models/model2.xml");
  

  ////net3.showInfo();
 

  NEUNet net4;
  net4.loadMod("./models/model2.xml");
  net4.test(testInput, testTarget);
  //cv::Mat maxhere = net4.predict(input);
  //
  //std::cout << "maxhere:" << std::endl;
  //std::cout << maxhere << std::endl;
  return 0;
}



