#include <stdio.h>
#include <iostream>
#include "cublas_v2.h"
#include "debug.h"
#include "cudnn_conv.h"

const int WARP_SIZE{32};

const bool COMPARE_WITH_CPU{false};
const bool COMPARE_WITH_CUDNN{true};

const int BATCH_SIZE{128};
const int IN_X_SIZE{128};
const int IN_Y_SIZE{128};
const int IN_CHANNELS{128};
const int IN_SIZE{IN_X_SIZE * IN_Y_SIZE * IN_CHANNELS * BATCH_SIZE};
const int IN_BYTES{IN_SIZE * sizeof(float)};
const int FILTER_X_SIZE{3};
const int FILTER_Y_SIZE{3};
const int OUT_CHANNELS{128};
const int OUT_X_SIZE{IN_X_SIZE - FILTER_X_SIZE + 1};
const int OUT_Y_SIZE{IN_Y_SIZE - FILTER_Y_SIZE + 1};
const int OUT_SIZE{OUT_X_SIZE * OUT_Y_SIZE * OUT_CHANNELS * BATCH_SIZE};
const int OUT_BYTES{OUT_SIZE * sizeof(float)};

// 32 * 32 = 1024, the max threads per block.
const int TILE_X_SIZE{32};
const int TILE_Y_SIZE{32};

const int WINDOW_X_SIZE = TILE_X_SIZE + FILTER_X_SIZE - 1;
const int WINDOW_Y_SIZE = TILE_Y_SIZE + FILTER_Y_SIZE - 1;
// WINDOW_X_SIZE rounded up to nearest multiple of WARP_SIZE
// TODO should this be computed in kernel, so it can be smaller for possible edge-cases of small windows at
// right and/or bottom of image?
const int ROUNDED_WINDOW_X_SIZE = ((WINDOW_X_SIZE - 1) / WARP_SIZE + 1) * WARP_SIZE;
const int ROUNDED_WINDOW_Y_SIZE = ((WINDOW_Y_SIZE - 1) / WARP_SIZE + 1) * WARP_SIZE;

const int TILES_ALONG_X = (OUT_X_SIZE - 1) / TILE_X_SIZE + 1;
const int TILES_ALONG_Y = (OUT_Y_SIZE - 1) / TILE_Y_SIZE + 1;

__host__ __device__ int i_ind(const int b, const int c, const int y, const int x) {
  return b * IN_X_SIZE * IN_Y_SIZE * IN_CHANNELS + c * IN_X_SIZE * IN_Y_SIZE + y * IN_X_SIZE + x;
}
__host__ __device__ int o_ind(const int b, const int c, const int y, const int x) {
  return b * OUT_X_SIZE * OUT_Y_SIZE * OUT_CHANNELS + c * OUT_X_SIZE * OUT_Y_SIZE + y * OUT_X_SIZE + x;
}
__host__ __device__ int f_ind(const int i_c, const int o_c, const int y, const int x) {
  return o_c * FILTER_X_SIZE * FILTER_Y_SIZE * IN_CHANNELS + i_c * FILTER_X_SIZE * FILTER_Y_SIZE + y * FILTER_X_SIZE + x;
}
__host__ __device__ int w_ind(const int c, const int y, const int x) {
  return c * TILE_X_SIZE * TILE_Y_SIZE + y * TILE_X_SIZE + x;
}


// Computes convolution and saves output. 
void host_conv(float const * const im, float const * const filter, float * out) {
  for( int b = 0; b < BATCH_SIZE; b++) {
    for( int o_x = 0; o_x < OUT_X_SIZE; o_x++) {
      for( int o_y = 0; o_y < OUT_Y_SIZE; o_y++) {
        for( int o_c = 0; o_c < OUT_CHANNELS; o_c++) {
          float temp = 0.0;
          for( int f_x = 0; f_x < FILTER_X_SIZE; f_x++) {
            for( int f_y = 0; f_y < FILTER_Y_SIZE; f_y++) {
              for( int i_c = 0; i_c < IN_CHANNELS; i_c++) {
                temp += (filter[f_ind(i_c, o_c, f_y, f_x)] * 
                  im[i_ind(b, i_c, o_y + f_y, o_x + f_x)]);
              }
            }
          }
          out[o_ind(b, o_c, o_y, o_x)] = temp;
        }
      }
    }
  }
}

/**
  Indexing assumptions:
  filter is formatted (in_channels, out_channels, height, width)
  image is formatted (in_channels, height, width)
  note that this means for im[c][y][x] the lowest order index, the x index, comes last
  same for filter: filter[ic][oc][y][x]: x is the lowest index
*/
__global__ void gpu_conv(float const * const im, float const * const filter, float * out) {
  // shared memory for filter and for the relevant window of the image
  // shared memory need sto contain filter of size FILTER_X_SIZE x FILTER_Y_SIZE,
  // and window of image that we are convolving, of size
  // (WINDOW_X_SIZE, WINDOW_Y_SIZE)
  extern __shared__ float shared[];
  float* fs{&shared[0]};
  float* window{&shared[FILTER_X_SIZE * FILTER_Y_SIZE]};

  // the upper-left index of the tile we are processing
  // note that these indices can be used to index both the upper-left of the output tile,
  // and the upper-left of the input window 

  const int tile_x = (blockIdx.x % TILES_ALONG_X) * TILE_X_SIZE;
  const int tile_y = (blockIdx.x / TILES_ALONG_X) * TILE_Y_SIZE;

  // keeps track of accumulated output for this thread
  float acc = 0;

  const int batch_index = blockIdx.y;
  const int o_c = blockIdx.z;
  const int o_x = tile_x + (threadIdx.x % TILE_X_SIZE);
  const int o_y = tile_y + (threadIdx.x / TILE_X_SIZE);

  const int valid_window_x_size = min(WINDOW_X_SIZE, IN_X_SIZE - tile_x);
  const int valid_window_y_size = min(WINDOW_Y_SIZE, IN_Y_SIZE - tile_y);

  const int tile_x_size = valid_window_x_size - FILTER_X_SIZE + 1;
  const int tile_y_size = valid_window_y_size - FILTER_Y_SIZE + 1;


  // possible idea: have output tile be in shared memory also, to make
  // accumulation over multiple timesteps and multiple locations easier


  //if (blockIdx.x == 3 && threadIdx.x == 0) {
  //        printf("TILES_ALONG_X,y: %d, %d\n", TILES_ALONG_X, TILES_ALONG_Y);
  //        printf("valid window sizes: %d, %d\n", valid_window_x_size, valid_window_y_size);
  //        printf("tile_x, tile_y: %d, %d\n", tile_x, tile_y);
  //        printf("tile sizes: %d, %d\n", tile_x_size, tile_y_size);
  //        printf("threadIdx.x: %d\n", threadIdx.x);
  //}


  for (int i_c = 0; i_c < IN_CHANNELS; ++i_c) {

    // CURRENT PROBLEM: for each output channel, we load this all into memory AGAIN.
    // A location of the input is loaded into memory (num output channels) times.
    // AND after each time we have to call __syncthreads(). Not ideal. Let's try 
    // just updating all output channels for each filter 
          
    // load filter into shared memory, for a single input channel
    for (int index = threadIdx.x; index < FILTER_X_SIZE * FILTER_Y_SIZE; index += blockDim.x) {
      fs[index] = filter[o_c * IN_CHANNELS * FILTER_X_SIZE * FILTER_Y_SIZE +  i_c * FILTER_X_SIZE * FILTER_Y_SIZE + index];
      //if (i_c == 1 and index == 0) {
      //  printf("Setting fs[0]: %f\n", fs[index]);
      //  printf("IN_CHANNELS : %d\n", IN_CHANNELS);
      //}
    }

    // load the single-channel tile of the input into shared memory
    for (int index = threadIdx.x; index < ROUNDED_WINDOW_X_SIZE * valid_window_y_size; index += blockDim.x) {

      int w_x = index % ROUNDED_WINDOW_X_SIZE;
      int w_y = (index / ROUNDED_WINDOW_X_SIZE); 
      int i_x = tile_x + w_x;
      int i_y = tile_y + w_y;

      if (i_x < IN_X_SIZE && i_y < IN_Y_SIZE && w_x < valid_window_x_size && w_y < valid_window_y_size) {
        window[w_y * WINDOW_X_SIZE + w_x] = im[i_ind(batch_index, i_c, i_y, i_x)];
      }
    }

    // TODO the largest "sampling data (not issued)" entry in the nsight-compute source profiler
    // is this barrier (it shows up as a barrier on the previous instruction, the above for loop).
    // Ways to reduce: process as many in_channels as possible. Can utilize much more of the
    // shared memory.
    __syncthreads();
    
    // loop over filter to accumulate convolution at thread-specific output location
    // TODO ordering of these loops?
    
    for(int f_x = 0; f_x < FILTER_X_SIZE; ++f_x) {
      for(int f_y = 0; f_y < FILTER_Y_SIZE; ++f_y) {
        // compute indices into the shared window
        int w_x = threadIdx.x % TILE_X_SIZE  + f_x;
        int w_y = threadIdx.x / TILE_X_SIZE + f_y;
        //if (o_x == testx && o_y == testy) {
        //        //printf("F: %d,%d: %f\n", f_y, f_x, fs[f_y * FILTER_Y_SIZE + f_x]);
        //        printf("W: %d,%d: %f\n", w_x, w_y, window[w_y * WINDOW_X_SIZE + w_x]);
        //}

        if (w_y < valid_window_y_size && w_x < valid_window_x_size) {
          acc += (fs[f_y * FILTER_X_SIZE + f_x] * 
              // for the window, since w_y is the lowest index, make sure that w_y varies along threadIdx.x
              window[w_y * WINDOW_X_SIZE + w_x]);
        }
      }
    }
    // need to sync before we overwrite filter and window again.
    __syncthreads();
  }

  if (o_x - tile_x < tile_x_size && o_y - tile_y < tile_y_size) {
    out[o_ind(batch_index,o_c,o_y,o_x)] = acc;
  }
}



int main( int argc, char *argv[] )
{

  /* get GPU device number and name */
  int dev;
  cudaDeviceProp deviceProp;
  checkCUDA( cudaGetDevice( &dev ) );
  checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
  printf("Using GPU %d: %s\n", dev, deviceProp.name );

  fprintf(stdout, "\nImage size is %d x %d x %d\n",IN_CHANNELS, IN_X_SIZE, IN_Y_SIZE);
  fprintf(stdout, "Filter size is %d x %d x %d x %d\n",IN_CHANNELS, OUT_CHANNELS, FILTER_X_SIZE, FILTER_Y_SIZE);
  fprintf(stdout, "Batch size is %d\n",BATCH_SIZE);

  // arrays for image and filter on host and device, and
  // output from cpu and from gpu
  float *h_image, *h_filter, *h_gpu_out, *h_cpu_out;
  float *d_image, *d_filter, *d_out;

  // allocate space for image and convolutional filter
  cudaError_t status;
  status = cudaMallocHost((void**) &h_image, IN_BYTES);
  if (status != cudaSuccess) {
    printf("Error allocating image memory.\n");
  }
  const int filter_size = FILTER_X_SIZE * FILTER_Y_SIZE * IN_CHANNELS * OUT_CHANNELS;
  const int filter_bytes = sizeof(float) * filter_size;
  status = cudaMallocHost((void**) &h_filter, filter_bytes);
  if (status != cudaSuccess) {
    printf("Error allocating filter memory.\n");
  }

  // allocate space for output from both CPU and GPU
  h_cpu_out = (float *) malloc(OUT_BYTES);
  status = cudaMallocHost((void**) &h_gpu_out, OUT_BYTES);
  if (status != cudaSuccess) {
    printf("Error allocating ouput memory.\n");
  }

  // check if there was a error in host malloc
  if( h_image == NULL || h_filter == NULL || h_cpu_out == NULL ||
      h_gpu_out == NULL) {
    fprintf(stderr,"Error in host malloc\n");
    return 911;
  }

  // initialize output to 0 (since we will be computing it by adding to this
  // output
  memset( h_cpu_out, 0, OUT_BYTES);
  memset( h_gpu_out, 0, OUT_BYTES);

  // randomly initialize the image and the filter
  // TODO redo random init
  for( int i = 0; i < IN_SIZE; ++i ) {
    h_image[i] = float( rand() ) / ( float(RAND_MAX) + 1.0 );
    //h_image[i] = 1.0 + ((i / (IN_X_SIZE * IN_Y_SIZE)) % 2);
    //h_image[i] = 1.0;
  }
  for(int i = 0; i < filter_size; ++i ) {
    h_filter[i] = float( rand() ) / ( float(RAND_MAX) + 1.0 );
    //h_filter[i] = 0.0 + ((i / (IN_X_SIZE * IN_Y_SIZE)) % 2);
    //h_filter[i] = 1.0;
  }


  // start timers
  cudaEvent_t start, stop;
  checkCUDA( cudaEventCreate( &start ) );
  checkCUDA( cudaEventCreate( &stop ) );
  checkCUDA( cudaEventRecord( start, 0 ) );

  /////////
  // GPU //
  /////////

  // allocate image, filter, and output in GPU memory
  checkCUDA( cudaMalloc( (void **)&d_image, IN_BYTES ) );
  checkCUDA( cudaMalloc( (void **)&d_filter, filter_bytes ) );
  checkCUDA( cudaMalloc( (void **)&d_out, OUT_BYTES ) );
  // copy image and filter to device
  checkCUDA( cudaMemcpy( d_image, h_image, IN_BYTES, cudaMemcpyHostToDevice ) );
  checkCUDA( cudaMemcpy( d_filter, h_filter, filter_bytes, cudaMemcpyHostToDevice ) );

  // setup grid and block sizes
  const dim3 threads(1024, 1, 1); // max possible for me. one thread per entry in 32x32 tile 
  const dim3 blocks( TILES_ALONG_X * TILES_ALONG_Y, BATCH_SIZE, OUT_CHANNELS);

  const int window_size = WINDOW_X_SIZE * WINDOW_Y_SIZE;
  const int sharedMemSize = sizeof(float) * (FILTER_X_SIZE * FILTER_Y_SIZE + window_size);

  // start timer
  printf("ROUNDED_WINDOW_X_SIZE: %d\n", ROUNDED_WINDOW_X_SIZE);
  printf("ROUNDED_WINDOW_Y_SIZE: %d\n", ROUNDED_WINDOW_Y_SIZE);
  printf("WINDOW_X_SIZE: %d\n", WINDOW_X_SIZE);
  printf("WINDOW_Y_SIZE: %d\n", WINDOW_Y_SIZE);

  printf("blocks.z: %d\n", blocks.z);
  printf("blocks.x: %d\n", blocks.x);
  printf("blocks.y: %d\n", blocks.y);
  printf("blocks.z: %d\n", blocks.z);
  printf("threads: %d\n", threads.x);
  printf("filter_size: %d\n", filter_size);
  printf("window_size: %d\n", window_size);
  printf("shmem size: %d\n", sharedMemSize);

  checkCUDA( cudaEventRecord( start, 0 ) );

  // initialize output to 0, since compute it by adding to it. 
  // TODO: is this better than just having each thread initialize its loc to 0?
  checkCUDA( cudaMemset( d_out, 0, OUT_BYTES ) );

  //gpu_conv<<< blocks, threads, sharedMemSize >>> (d_image, d_filter, d_out);
  gpu_conv<<< blocks, threads, sharedMemSize >>> (d_image, d_filter, d_out);
  checkKERNEL();

  // stop timer and print time
  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  float elapsedTime;
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  fprintf(stdout, "Total time GPU is %f sec\n", elapsedTime / 1000.0f );
  fprintf(stdout, "Performance is %f GFlop/s\n", ( ( (double) BATCH_SIZE *
    (double) OUT_X_SIZE * (double) OUT_Y_SIZE * 2.0 * (double) FILTER_X_SIZE *
    (double) FILTER_Y_SIZE * (double) IN_CHANNELS * (double) OUT_CHANNELS) / 
    ( (double) elapsedTime / 1000.0 ) * 1.e-9 ));

	
  // copy output back to host to compare with CPU
  checkCUDA( cudaMemcpy( h_gpu_out, d_out, OUT_BYTES, cudaMemcpyDeviceToHost ) );

  if (COMPARE_WITH_CPU) {
    // start timer
    checkCUDA( cudaEventRecord( start, 0 ) );
  
    // do convolution on cpu
    host_conv(h_image, h_filter, h_cpu_out);
  
    // stop timers
    checkCUDA( cudaEventRecord( stop, 0 ) );
    checkCUDA( cudaEventSynchronize( stop ) );
    checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );
  
    // print time taken
    fprintf(stdout, "Total time CPU is %f sec\n", elapsedTime / 1000.0f );
    checkCUDA( cudaEventDestroy( start ) );
    checkCUDA( cudaEventDestroy( stop ) );
  
    // compare GPU implementation results with CPU results
    float temp = 0.0;
    for( int i = 0; i < OUT_SIZE; i++ )
    {
      //printf("CPU : %f, GPU : %f\n", h_cpu_out[i], h_gpu_out[i]);
      temp += ( h_cpu_out[i] - h_gpu_out[i] ) * ( h_cpu_out[i] - h_gpu_out[i] );
    } /* end for */

    printf("error is %f\n",temp);
    if( temp > 10 ) printf("FAIL\n");
    else printf("PASS\n");
  }

  if (COMPARE_WITH_CUDNN) {
    float* cudnn_output; float* h_cudnn_out;
    checkCUDA( cudaMalloc( (void **)&cudnn_output, OUT_BYTES ));

    cudaMallocHost((void**) &h_cudnn_out, OUT_BYTES);
    if (status != cudaSuccess) {
      printf("Error allocating cudnn host memory.\n");
    }

    float cudnn_time;

    cudnn_time = cudnnConv(d_image, d_filter, cudnn_output, IN_X_SIZE, IN_Y_SIZE, IN_CHANNELS, BATCH_SIZE, OUT_CHANNELS, FILTER_X_SIZE, FILTER_Y_SIZE, OUT_X_SIZE, OUT_Y_SIZE);


    checkCUDA( cudaMemcpy( h_cudnn_out, cudnn_output, OUT_BYTES, cudaMemcpyDefault ) );


    // compare CUDNN implementation results with CPU results
    float temp = 0.0;
    for( int i = 0; i < OUT_SIZE; i++ ) {
      //printf("CPU : %f, GPU : %f\n", h_cpu_out[i], h_gpu_out[i]);
      float orig_temp = temp;
      temp += ( h_gpu_out[i] - h_cudnn_out[i] ) * ( h_gpu_out[i] - h_cudnn_out[i] );
    } 

    printf("CUDNN error from GPU is %f\n",temp);
    if( temp > 10 ) printf("CUDNN FAIL\n");
    else printf("CUDNN PASS\n");

    printf("CUDNN TIME: %f", cudnn_time);
    cudaFree(cudnn_output);
    cudaFreeHost(h_cudnn_out);
  }
    
    


  //int testx = 10;
  //int testy = 29;

  //printf("test index: %d, %d\n", testx, testy);
  //printf("cpu out: %f\n", h_cpu_out[testy * OUT_Y_SIZE + testx]);
  //printf("gpu out: %f\n", h_gpu_out[testy * OUT_Y_SIZE + testx]);


  // cleanup
  cudaFreeHost( h_image );
  cudaFreeHost( h_filter );
  free( h_cpu_out );
  cudaFreeHost( h_gpu_out );

  cudaFree(d_out);
  cudaFree(d_image);
  cudaFree(d_filter);

  cudaError_t cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceReset failed!");
      return 1;
  }

  return 0;


}
