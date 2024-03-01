#pragma once

#include"ops/normal.h"
#include"./ops/infer.h"

class Resnet
{
public:

  DataStream<float> run(DataStream<float>& input);
  void printRes(DataStream<float>& output);

};




