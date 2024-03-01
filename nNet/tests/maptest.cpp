#include<iostream>
#include<map>


using namespace std;

map<string, int> atRWstr = {
  {"SigAT", 0}, { "TanhAT",1 }, { "ReluAT",2 }, {"NullAT",3 }
};

static inline const std::string activTypeRW(int type)
{
  for (auto& tmp : atRWstr) {
    if (tmp.second == type)
      return tmp.first;
  }
  return string("");
}

int main(void) 
{
  //std::cout << activTypeRW(0) << std::endl;
  std::cout << atRWstr["SigAT"] << std::endl;
  return 0;
}







