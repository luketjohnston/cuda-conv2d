#ifndef __LUKE_CUDNN_CONV__
#define __LUKE_CUDNN_CONV__
#include <iostream>

/**
http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
this helped me get started with cudnn 
*/
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

float cudnnConv(float* input, float* filter, float* output, int in_height, int in_width, int in_channels, int batch_size, int out_channels, int kernel_height, int kernel_width, int out_height, int out_width);


#endif
