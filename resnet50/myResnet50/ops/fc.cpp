#include"fc.h"










//int mainfc11(void)
//{
//  vector<vector<int>> data1 = {
//  {1,2,3},
//  {4,5,6},
//  {7,8,9},
//  {10,11,12}
//  };
//  printMat2D(data1);
//  vector<vector<int>> data2 = mat2DTranspose(data1);
//  printMat2D(data2);
//
//  vector<vector<int>> data5 = mat2DMul(data1, data2);
//  printMat2D(data5);
//
//  int* data3;
//  vvt2t(data2, &data3);
//  
//  printMat2D(data3, data2[0].size(), data2.size());
//
//  unsigned int h4, w4;
//  int* data4 = mat2DTranspose(data3, data2.size(), data2[0].size(), &h4, &w4);
//  printMat2D(data4, w4, h4);
//
//  int res = mat2DHWMul(data3, data4, 4, 3, 3, 4, 0, 0);
//  printf("\nres=%d\n", res);
//
//  int* data6 = mat2DMul(data3, data4, 4, 3, 3, 4);
//  printMat2D(data6, 3, 3);
//
//  delete[] data3;
//  delete[] data4;
//  delete[] data6;
//  return 0;
//}


//int mainfc13131(void)
//{
//  vector<vector<int>> data1 = {
//  {1,2,3},
//  {4,5,6},
//  {7,8,9},
//  {10,11,12}
//  };
//
//  vector<vector<int>> data2;
//  //data2 = mat2DAdd(data1, data1);
//  //printMat2D(data2);
//
//  int* data3;
//  vvt2t(data1, &data3);
//  printMat2D(data3, 3, 4);
//  int* data4 = mat2DAdd(data3, data3, 4, 3);
//  printMat2D(data4, 3, 4);
//  return 0;
//}



//int mainada231(void) 
//{
//  vector<vector<int>> data1 = {
//  {1,2,3},
//  {4,5,6},
//  {7,8,9},
//  {10,11,12}
//  };
//
//  vector<vector<int>> w = {
//    {1,0,0},
//    {0,1,0},
//    {0,0,1},
//  };
//
//  vector<vector<int>> b = {
//    {100},
//    {100},
//    {100},
//  };
//
//  //vector<vector<int>> xs = mat2DTranspose(data1);
//  //printMat2D(xs);
//  vector<vector<int>> y;
//  for (unsigned int h_ = 0; h_ < data1.size(); ++h_) {
//    vector<vector<int>> x;
//    x.push_back(data1[h_]);
//    
//    x = mat2DTranspose(x);
//    printMat2D(x);
//    x=fc2D(x,w,b);
//    //printMat2D(x);
//    x = mat2DTranspose(x);
//    y.push_back(x[0]);
//  }
//  //y=mat2DTranspose(y);
//  printMat2D(y);
//
//  return 0;
//}






