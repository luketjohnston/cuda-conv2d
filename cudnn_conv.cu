/**
http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
this helped me get started with cudnn 
*/


#include <cudnn.h>
#include "cudnn_conv.h"
#include "debug.h"
#include <iostream>


float cudnnConv(float* input, float* filter, float* output, int in_height, int in_width, int in_channels, int batch_size, int out_channels, int kernel_height, int kernel_width, int out_height, int out_width) {
  printf("here 0\n");
  cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));

  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW, // can also be CUDNN_TENSOR_NHWC
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/batch_size,
                                      /*channels=*/in_channels,
                                      /*image_height=*/in_height,
                                      /*image_width=*/in_width));

  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*out_channels=*/out_channels,
                                      /*in_channels=*/in_channels,
                                      /*kernel_height=*/kernel_height,
                                      /*kernel_width=*/kernel_width));

  // TODO fix filter formatting, the kind we use isn't an option for cudnn
  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW, // this means output_channels x input_channels x height x width
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/batch_size,
                                      /*channels=*/out_channels,
                                      /*image_height=*/out_height,
                                      /*image_width=*/out_width));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                           /*pad_height=*/0,
                                           /*pad_width=*/0,
                                           /*vertical_stride=*/1,
                                           /*horizontal_stride=*/1,
                                           /*dilation_height=*/1,
                                           /*dilation_width=*/1,
                                           /*mode=*/CUDNN_CROSS_CORRELATION,
                                           /*computeType=*/CUDNN_DATA_FLOAT));

  cudnnConvolutionFwdAlgoPerf_t convolution_algorithm_perf;
  int returnedAlgoCount;
  checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
                                        input_descriptor,
                                        kernel_descriptor,
                                        convolution_descriptor,
                                        output_descriptor,
                                        1, // requested algo count
                                        &returnedAlgoCount,
                                        &convolution_algorithm_perf));
  cudnnConvolutionFwdAlgo_t convolution_algorithm = convolution_algorithm_perf.algo;

  size_t workspace_bytes = 0;
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   convolution_descriptor,
                                                   output_descriptor,
                                                   convolution_algorithm,
                                                   &workspace_bytes));

  void* d_workspace{nullptr};
  checkCUDA(cudaMalloc(&d_workspace, workspace_bytes));

  float elapsedTime; 
  cudaEvent_t start, stop;
  checkCUDA( cudaEventCreate( &start ) );
  checkCUDA( cudaEventCreate( &stop ) );

  checkCUDA( cudaEventRecord( start, 0 ) );

  const float alpha = 1, beta = 0;
  checkCUDNN(cudnnConvolutionForward(cudnn,
                                   &alpha,
                                   input_descriptor,
                                   input,
                                   kernel_descriptor,
                                   filter,
                                   convolution_descriptor,
                                   convolution_algorithm,
                                   d_workspace,
                                   workspace_bytes,
                                   &beta,
                                   output_descriptor,
                                   output));
  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  return elapsedTime / 1000.0f;
}

