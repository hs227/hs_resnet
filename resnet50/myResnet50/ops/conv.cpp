#include"conv.h"





 
//int main(void) 
//{
//  const unsigned int lenY = 9;
//  const unsigned int lenX = 9;
//  float tt[lenY][lenX];
//  //float tt1[3][3];
//  float* tt1 = new float[3 * 3];
//  for (int y = 0; y < 3; ++y) {
//    for (int x = 0; x < 3; ++x) {
//      *(tt1+y*3+x) = 2;
//    }
//  }
//  for (int y = 0; y < lenY; ++y) {
//    for (int x = 0; x < lenX; ++x) {
//      tt[y][x] = y * 10 + x;
//    }
//  }
//  printMat2D<float>((float*)tt, (const unsigned int)lenX, (const unsigned int)lenY);
//  std::cout<<std::endl;
//  //float a33=mat2DWindowElem<float>((float*)tt, (const unsigned int)lenX, (const unsigned int)lenY,
//  //  0, 0, 3, 3);
//  //printf("%f\n", a33);
//  //
//  //mat2DWindow<float>((float*)tt, (const unsigned int)lenX, (const unsigned int)lenY, (float*)tt1, 3, 3, 3, 3);
//  //printMat2D<float>((float*)tt1, 3,3);
//  //for (int y = 0; y < lenY; ++y) {
//  //  for (int x = 0; x < lenX; ++x) {
//  //    tt[y][x] =  0;
//  //  }
//  //}
//  //printMat2D<float>((float*)tt, lenX, lenY);
//  //mat2DWindowT<float>((float*)tt, 9, 9, (float*)tt1, 3, 0, 3, 3);
//  //printMat2D<float>((float*)tt, lenX,lenY);
//
//  //float* window=convTraversal<float>((float*)tt, lenX, lenY, 3, 3,TraversalFlag::BIG);
//  //printMat2D<float>((float*)window, 21, 21);
//  //delete[] window;
//  float res = convMul<float>((float*)tt1, (float*)tt1, 3, 3);
//  printf("res=%f\n", res);
//  delete[] tt1;
//  return 0;
//}


//int main31(void) 
//{
//  int input[5][5] = {
//    3,3,2,1,0,
//    0,0,1,3,1,
//    3,1,2,2,3,
//    2,0,0,2,2,
//    2,0,0,0,1,
//  };
//  int weight[3][3] = {
//    0,1,2,
//    2,2,0,
//    0,1,2,
//  };
//
//  //int input1[3][3];
//  //mat2DWindow<int>((int*)input, 5, 5, (int*)input1, 0, 0, 3, 3);
//  //printMat2D<int>((int*)input1, 3, 3);
//  //int res = conv2DMul<int>((int*)input1, (int*)weight, 3, 3);
//  //printf("%d\n", res);
//
//  unsigned int padding[4] = { 0,0,0,0 };
//  unsigned int strider[2] = { 2,2 };
//  unsigned int new_width, new_height;
//  int* output = conv2D<int>((int*)input, 5, 5, (int*)weight, 3, 3, padding,strider,&new_width,&new_height);
//  printMat2D<int>((int*)output, new_width, new_height);
//  delete[] output;
//
//  //int* padding_out;
//  //unsigned int new_height;
//  //unsigned int new_width;
//  //padding_out = mat2DPadding<int>((int*)input, 5, 5, 1, 1, 1, 1, &new_width, &new_height);
//  //printMat2D<int>((int*)padding_out, new_width, new_height);
//  //delete[] padding_out;
//
//  return 0;
//}



//int mainra(void) 
//{
//  vector<vector<int>> input = {
//    {3,3,2,1,0},
//    {0,0,1,3,1},
//    {3,1,2,2,3},
//    {2,0,0,2,2},
//    {2,0,0,0,1},
//  };
//
//  printMat2D(input);
//  int a=mat2DWindowElem(input, 0, 0, 3, 0);
//  printf("a=%d\n", a);
//  vector<vector<int>> padded = mat2DPadding(input, 1, 1, 1, 1);
//  printMat2D(padded);
//  vector<vector<int>> input1 = {
//    {1,0,0},
//    {0,1,0},
//    {0,0,1},
//  };
//  //mat2DWindow(input, input1, 0, 0, 3, 3);
//  //printMat2D(input1);
//  //mat2DWindowT(input, input1, 0, 0, 3, 3);
//  //printMat2D(input);
//  //vector<vector<int>> window = convTraversal(input, 3, 3, TraversalFlag::BIG);
//  //printMat2D(window);
//  return 0;
//}
