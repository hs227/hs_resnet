#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


#include "./resnet.h"
#include"ops/infer.h"

#include <filesystem> // C++17 standard header file name

using namespace std;


// Renset50中每张图片为HWC=224*224*3


#define BOOLSTR(x) ((x)?"True":"False")


/* 获得所有图片的文件路径 */
vector<string> getFileName(void)
{
  vector<string> filenames;
  static string dir_path("./src/resnet50/pics/ani_12/");
  filenames.push_back(dir_path + "Gou.jpg");
  return filenames;
  for (const filesystem::directory_entry& entry : filesystem::directory_iterator(dir_path))
  {
    if (entry.is_regular_file()) {
      std::cout << entry.path() << " :-> " << entry.path().filename() << std::endl;
      filenames.push_back(entry.path().string());
    }

  }
  return filenames;
}


/* 对于单张图片的预处理 */
float* preprocessImg(const std::string& file_name) {
  auto transposed = [](float* in, float* out,
    int zi, int yi, int xi,
    int zo, int yo, int xo) {
      //memcpy(out, in, zo * yo * xo);
      //return;
      // HWC->CWH
      for (int z_ = 0; z_ < zi; z_++) {
        for (int y_ = 0; y_ < yi; y_++) {
          for (int x_ = 0; x_ < xi; x_++) {
            //out[x][y][z] = in[z][y][x];
            *(out + x_ * yo * xo + y_ * xo + z_) = *(in + z_ * yi * xi + y_ * xi + x_);
          }
        }
      }
  };

  float* mat_data = (float*)malloc(224 * 224 * 3 * sizeof(float));
  cv::Mat source_o, img_o, img_r;

  source_o = cv::imread(file_name);
  cv::cvtColor(source_o, img_o, cv::COLOR_BGR2RGB);
  cv::resize(img_o, img_r, { 224, 224 });

  uint8_t* trans_data = (uint8_t*)malloc(224 * 224 * 3 * sizeof(uint8_t));
  if (trans_data == 0) { exit(EXIT_FAILURE); }
  memcpy(trans_data, (uint8_t*)img_r.data, 224 * 224 * 3);

  for (int i = 0; i < 224; i++) {
    for (int j = 0; j < 224; j++) {
      mat_data[i * 224 * 3 + j * 3 + 0] =
        ((trans_data[i * 224 * 3 + j * 3 + 0] / 255.0) - 0.485) / 0.229;  // R
      mat_data[i * 224 * 3 + j * 3 + 1] =
        ((trans_data[i * 224 * 3 + j * 3 + 1] / 255.0) - 0.456) / 0.224;  // G
      mat_data[i * 224 * 3 + j * 3 + 2] =
        ((trans_data[i * 224 * 3 + j * 3 + 2] / 255.0) - 0.406) / 0.225;  // B
    }
  }
  // NCHW -> NHWC
  float* matt_data = (float*)malloc(3 * 224 * 224 * sizeof(float));
  transposed(mat_data, matt_data, 224, 224, 3, 3, 224, 224);

  free(mat_data);
  free(trans_data);

  return matt_data;
}



/* 获得时间 */
long long getTime()
{
  long long timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
  return timestamp;
}

struct BatchFile {
  vector<string> filenames;
};

/* 将文件按batch分割 */
vector<BatchFile> batchFiles(vector<string> filenames, const int batch_num) 
{
  vector<BatchFile> batches;
  const int file_num = filenames.size();
  BatchFile one_batch;
  for (int i = 0; i < file_num; ++i) {
    one_batch.filenames.push_back(filenames[i]);
    if ((i+1) % batch_num == 0) {
      batches.push_back(one_batch);
      one_batch.filenames.clear();
    }
  }
  if (!one_batch.filenames.empty()) {
    batches.push_back(one_batch);
  }
  return batches;
}
void printBatchFiles(vector<BatchFile> batches) 
{
  const int batch_num = batches.size();
  for (int ib = 0; ib < batch_num; ++ib) {
    printf("batch[%d]:\n", ib);
    const int file_num = batches[ib].filenames.size();
    for (int if_ = 0; if_ < file_num; if_++) {
      printf("   %s\n", batches[ib].filenames[if_].c_str());
    }
  }
}


bool checkEquDSF(vector<float*> old_datas, DataStream<float>& new_data) 
{
  const unsigned int n = new_data.n;
  const unsigned int h = new_data.height;
  const unsigned int w = new_data.width;
  const unsigned int c = new_data.channel;

  for (unsigned int in = 0; in < n; ++in) {
    float* old_data = old_datas[in];
    for (unsigned int ic = 0; ic < c; ++ic) {
      for (unsigned int ih = 0; ih < h; ++ih) {
        //int chk = memcmp(&old_data[ic * h * w + ih * w], &new_data.datas[in].data[ic][ih][0], sizeof(float) * w);
        //if (chk != 0)
        //  return false;
        for (unsigned int iw = 0; iw < w; ++iw) {
          if (old_data[ic * h * w + ih * w + iw] != new_data.datas[in].data[ic][ih][iw]) {
            return false;
          }
        }
      }
    }
  }
  return true;
}



DataStream<float>& preprocess(BatchFile& batch) 
{
  const int batch_size = batch.filenames.size();
  vector<float*> batch_data;
  for(int i=0;i<batch_size;++i)
  {
    float* img = preprocessImg(batch.filenames[i]);
    batch_data.push_back(img);
  }
  DataStream<float> stream=DataStream<float>(batch_data,224,224,3);
  //{
  //  bool isequed=checkEquDSF(batch_data, stream);
  //  printf("Equed>?%s\n", BOOLSTR(isequed));
  //  vector<float*> newi = stream.vout(NULL,NULL,NULL,NULL);
  //  isequed = checkEquDSF(newi, stream);
  //  printf("Equed<?%s\n", BOOLSTR(isequed)); 
  //}
  for (int i = 0; i < batch_size; ++i) {
    free(batch_data[i]);
  }

  return stream;
}




int main() {
  vector<string> files = getFileName();
  if (files.empty()) {
    return 0;
  }
  
  // 单次batch的图片数
  int batch_num = 4;
  vector<BatchFile> batch_names = batchFiles(files, batch_num);
  printBatchFiles(batch_names);
  //return 0;


  long long total_time = 0;
  Resnet resnet50;

  const int batches_size = batch_names.size();
  for (int ib = 0; ib < batches_size; ++ib) 
  {
    long long start = getTime();
    DataStream<float> dataStreams = preprocess(batch_names[ib]);
    dataStreams=resnet50.run(dataStreams);
    long long end = getTime();
    long long time = end - start;
    total_time += time;
    std::cout << "\033[0;32mTime cost : " << time << " ms.\033[0m" << std::endl;
    
    resnet50.printRes(dataStreams);
  }
  float latency = (float)(total_time) / (float)(files.size());
  std::cout << "\033[0;32mAverage Latency : " << latency << "ms \033[0m" << std::endl;
  std::cout << "\033[0;32mAverage Throughput : " << (1000 / latency) << "fps \033[0m" << std::endl;

  /*long long total_time = 0;
  Resnet resnet50;
  for (string& it : files) {
    long long start = getTime();
    std::cout << "Predict : " << it << std::endl;
    float* img = preprocess(it);

    img = resnet50.run(img);

    long long end = getTime();
    long long time = end - start;
    total_time += time;
    std::cout << "\033[0;32mTime cost : " << time << " ms.\033[0m" << std::endl;
    show_res(img);
    free(img);
  }

  float latency = (float)(total_time) / (float)(files.size());
  std::cout << "\033[0;32mAverage Latency : " << latency << "ms \033[0m" << std::endl;
  std::cout << "\033[0;32mAverage Throughput : " << (1000 / latency) << "fps \033[0m" << std::endl;*/
  return 0;
}






