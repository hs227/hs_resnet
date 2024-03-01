#include<iostream>
#include<opencv2\opencv.hpp>
#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<vector>


void printMat(cv::Mat& matx) 
{
  std::cout << "[" << std::endl;
  for (int r_ = 0; r_ < matx.rows; ++r_) {
    for (int c_ = 0; c_ < matx.cols; ++c_) {
      std::cout << ", " << matx.at<float>(r_, c_);
    }
    std::cout << std::endl;
  }
  std::cout << "]" << std::endl;
}



cv::Mat maxLoc(cv::Mat& matx) 
{
  using namespace std;
  cv::Mat sorted_index,sorted_matx;
  cv::sortIdx(matx, sorted_index, 
    cv::SORT_EVERY_COLUMN +cv::SORT_DESCENDING);
  cv::sort(matx, sorted_matx,
    cv::SORT_EVERY_COLUMN + cv::SORT_DESCENDING);

  
  cv::Mat indexes = cv::Mat_<float>(sorted_index);
  cv::Mat maxloc = indexes.row(0);
 
  cout << "sorted_index:" << endl;
  cout << sorted_index << endl;
  cout << "sorted_matx:" << endl;
  cout << sorted_matx << endl;
  cout << "indexes:" << endl;
  cout << indexes << endl;
  cout << "maxloc:" << endl;
  cout << maxloc << endl;

  return maxloc;
}


int main(void)
{
  cv::Mat y_gred = (cv::Mat_<float>(4, 3)<<
    1,3,4,
    0,5,2,
    4,7,6,
    2,1,8);
  cv::Mat oneC = y_gred.col(0).clone();
  oneC.at<float>(3, 0) = 99;
  std::cout << "y_gred:" << std::endl;
  std::cout << y_gred << std::endl;
  //maxLoc(y_gred);

  

  std::cout << "oneC:" << std::endl;
  std::cout << oneC << std::endl;



  

  return 0;
}




