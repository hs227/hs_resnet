#include<iostream>
#include<vector>
#include<cmath>


// 一个简单的线性模型: y=a1*w1+a2*w2+a3*w3
double forward(double* a, double* w, int len) 
{
  double res = 0;
  for (int i = 0; i < len; ++i) {
    res += a[i] * w[i];
  }
  return res;
}

// 均方误差损失函数
double mse_loss(double y_pred, double y_true) 
{
  return pow(y_pred - y_true, 2);
}

// 均方误差损失导数
double d_mse_loss(double y_pred, double y_true)
{
  return 2 * (y_pred - y_true);
}

void backward(double* a,double* w, int len,
  double learning_rate, double y_true,double y_pred)
{
  double d_loss = d_mse_loss(y_pred, y_true);
  // update w
  for (int i = 0; i < len; ++i) {
    w[i] = w[i] - learning_rate * ( d_loss * a[i]);
  }
  
}


int main(void) 
{
  double a[3] = { 1,2,3 };
  double w[3] = { 0.1,0.1,0.1 };
  double learning_rate = 0.01;
  double y_true = 1;

  int max_deep = 1000;
  double min_loss = 1e-16;
  
  double loss = min_loss + 1;
  int deep;
  double y_pred=0;
  for (deep = 0; loss > min_loss && deep < max_deep; ++deep) 
  {
    y_pred= forward(a, w, 3);
    loss = mse_loss(y_pred, y_true);
    backward(a, w, 3, learning_rate,y_true,y_pred);
    printf("deep=%d\n", deep);
    printf("loss=%lf\n", loss);
    printf("y_true=%lf,y_pred=%lf\n", y_true, y_pred);
    printf("\n\n");
  }

  printf("End:\n");
  printf("w1=%lf,w2=%lf,w3=%lf\n", w[0], w[1], w[2]);
  return 0;
}

