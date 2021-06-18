#include <stdio.h>
#include "cublas_v2.h"
#include "../debug.h"

typedef float floatType_t;



#define COMPARE_WITH_CPU true

#define IN_IND(b, x, y, c, spec) ( ( (b) * (spec.in_x_size) * (spec.in_y_size) * (spec.in_channels)) + ( (c) * (spec.in_x_size) * (spec.in_y_size) ) + ( (y) * (spec.in_x_size) ) + (x) )
#define O_IND(b, x, y, c, spec) ( ( (b) * (spec.out_x_size) * (spec.out_y_size) * (spec.out_channels)) + ( (c) * (spec.out_x_size) * (spec.out_y_size) ) + ( (y) * (spec.out_x_size) ) + (x) )
#define F_IND(i_c, x, y, c, spec) ( ( (i_c) * (spec.filter_x_size) * (spec.filter_y_size) * (spec.filter_channels) ) + ( (c) * (spec.filter_x_size) * (spec.filter_y_size) ) + ( (y) * (spec.filter_x_size) ) + (x) )

#define W_IND(c, x, y, spec) ((c * spec.in_channels + x) * TILE_Y_SIZE + y) 

#define BATCH_SIZE (64)
//#define BATCH_SIZE (16)

#define IN_X_SIZE (128)
#define IN_Y_SIZE (128)
#define IN_CHANNELS (32)
#define IN_SIZE (IN_X_SIZE * IN_Y_SIZE * IN_CHANNELS * BATCH_SIZE)
#define IN_BYTES (IN_SIZE * sizeof(float))

#define FILTER_X_SIZE (9)// - 6)
#define FILTER_Y_SIZE (9)// - 6)
#define FILTER_CHANNELS (16)// - 12)
#define FILTER_SIZE (FILTER_X_SIZE * FILTER_Y_SIZE * FILTER_CHANNELS * IN_CHANNELS)
#define FILTER_BYTES (FILTER_SIZE * sizeof(float))

#define OUT_X_SIZE (IN_X_SIZE - FILTER_X_SIZE + 1)
#define OUT_Y_SIZE (IN_Y_SIZE - FILTER_Y_SIZE + 1)
#define OUT_CHANNELS (FILTER_CHANNELS)
#define OUT_SIZE (OUT_X_SIZE * OUT_Y_SIZE * OUT_CHANNELS * BATCH_SIZE)
#define OUT_BYTES (OUT_SIZE * sizeof(float))

// Thread dimensions
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32


struct conv_spec {
  int batch_size;
  int out_x_size;
  int out_y_size;
  int filter_channels;
  int filter_x_size;
  int filter_y_size;
  int in_channels;
}


// Computes convolution and saves output. Uses sizes defined above. 
// TODO: make sizes arguments
void host_conv(float const * const im, float const * const filter, float * out, conv_spec spec,) {
  for( int b = 0; b < spec.batch_size; b++) {
    for( int o_x = 0; o_x < spec.out_x_size; o_x++) {
      for( int o_y = 0; o_y < spec.out_y_size; o_y++) {
        for( int f_chan = 0; f_chan < spec.filter_channels; f_chan++) {
          float temp = 0.0;
          for( int f_x = 0; f_x < spec.filter_x_size; f_x++) {
            for( int f_y = 0; f_y < spec.filter_y_size; f_y++) {
              for( int i_chan = 0; i_chan < spec.in_channels; i_chan++) {
                temp += (filter[F_IND(i_chan, f_x,f_y,f_chan)] * 
                  im[IN_IND(b, o_x + f_x, o_y + f_y, i_chan)]);
              }
            }
          }
          out[O_IND(b, o_x,o_y,f_chan)] = temp;
        }
      }
    }
  }
}

__global__ void gpu_conv(float const * const im, float const * const filter, float * out, const conv_spec spec) {
  // shared memory for filter and for the relevant window of the image
  // shared memory need sto contain filter of size spec.filter_x_size x spec.filter_y_size,
  // and window of image that we are convolving, of size
  // (spec.threads_per_block_x + filter_x_size - 1) x (spec.threads_per_block_y + filter_y_size - 1)
  __shared__ float shared[];
  float fs[]{&shared};
  float window[]{&shared[spec.filter_x_size * spec.filter_y_size * spec.filter_channels]};

  // window_x_size is the size of the window loaded into shared memory
  // tile_x_size is the size of the output tile we are computing
  const int window_x_size = spec.tile_x_size + spec.filter_x_size - 1;
  const int window_y_size = spec.tile_y_size + spec.filter_y_size - 1;
  // maybe use these later
  //  int f_i = index % spec.in_channels;
  //  int f_x = (index / spec.in_channels) % spec.filter_x_size;
  //  int f_y = (index / spec.in_channels) / spec.filter_x_size;
  //  int f_c = (index / spec.in_channels) / spec.filter_x_size % spec.out_channels;
          
  // load filter into shared memory
  // TODO should fix this AFTER we know what order the filter is being iterated over in the main kernel loop below
  for (int index = threadIdx.x; index < spec.in_channels * spec.out_channels * spec.filter_x_size * spec.filter_y_size; index += blockDim.x) {
    fs[index] = filter[index];
  }


  // load window tile into shared memory
  for (int c = 0; c < spec.in_channels, ++c) {
    for (int off_x = 0; off_x < window_x_size; off_x += THREADS_PER_BLOCK_X) {
      for (int off_y = 0; off_y < window_y_size; off_y += THREADS_PER_BLOCK_Y) {
        // load into shared memory if necessary 
        int w_x = off_x + tix;
        int w_y = off_y + tiy;
        if (w_x < window_x_size && w_y < window_y_size) {
          if (o_x + off_x < spec.in_x_size && o_y + off_y < spec.in_y_size) {
            window[w_x][w_y] = im[IN_IND(b, o_x + off_x, o_y + off_y, i_chan)];
          }
        }
      }
  }

  __syncthreads();

  // each thread will process a single output entry at a time (meaning, one channel at
  // one location of the output image)

  // the upper-left index of the output tile we are processing
  const int tile_x = ((blockIdx.x * spec.tile_x_size) % spec.out_x_size); 
  const int tile_y = ((blockIdx.x * spec.tile_x_size) / spec.out_x_size) * spec.tile_y_size;
  // TODO any reason not to just use blockIdx.y for batch index?
  const int batch_index = blockIdx.y;

  // "index" is the index into the output tile (for the current thread). 
  for (index = threadIdx.x; index < spec.tile_x_size * spec.tile_y_size * spec.out_channels; index += blockDim.x) {

    int o_c = index % spec.out_channels;
    int o_x = tile_x + ((index / spec.out_channels) % spec.tile_x_size);
    int o_y = tile_y + ((index / spec.out_channels) / spec.tile_x_size);
          
    float acc = 0;
    for(f_x = 0; f_x < spec.filter_x_size; ++f_x) {
      for(f_y = 0; f_y < spec.filter_y_size; ++f_y) {
        // compute indices into the shared window
        int w_y = index % window_y_size + f_y;
        int w_x = (index / window_y_size) % window_x_size + f_x;
        for(f_i = 0; f_i < spec.in_channels; ++f_i) {

          // the filter access is the same for all threads in the block. So no need to worry about bank conflicts, etc.
          // only possible downside is that we're not fully utilizing shared memory thoroughput (instead of accessing 32 words
          // per warp per memory acccess, we only access 1). This seems unavoidable though.
          acc += (fs[F_IND(f_i,o_c,f_x,f_y)] * 
              // for the window, since w_y is the lowest index, make sure that w_y varies along threadIdx.x
              window[W_IND(f_i, w_x, w_y)];
        }
      }
    }
    out[O_IND(batch_index,o_x,o_y,o_c)] = acc;
    o_count += 1;
    o_c =  (blockIdx.x * blockDim.x + (threadIdx.x + blockDim.x * o_count)) % spec.out_channels;
    o_x = ((blockIdx.x * blockDim.x + (threadIdx.x + blockDim.x * o_count)) / spec.out_channels ) % spec.out_x_size;
    o_y = ((blockIdx.x * blockDim.x + (threadIdx.x + blockDim.x * o_count)) / spec.out_channels ) / spec.out_x_size;
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

  fprintf(stdout, "\nImage size is %d x %d x %d\n",IN_X_SIZE, IN_Y_SIZE, IN_CHANNELS);
  fprintf(stdout, "Filter size is %d x %d x %d\n",FILTER_X_SIZE, FILTER_Y_SIZE, FILTER_CHANNELS);
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
  status = cudaMallocHost((void**) &h_filter, FILTER_BYTES);
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
  for( int i = 0; i < IN_SIZE; i++ ) {
    h_image[i] = float( rand() ) / ( float(RAND_MAX) + 1.0 );
  }
  for(int i = 0; i < FILTER_SIZE; i++ ) {
    h_filter[i] = float( rand() ) / ( float(RAND_MAX) + 1.0 );
  }


  // First, we want to run the convolution on the CPU. 

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
  checkCUDA( cudaMalloc( (void **)&d_filter, FILTER_BYTES ) );
  checkCUDA( cudaMalloc( (void **)&d_out, OUT_BYTES ) );
  // copy image and filter to device
  checkCUDA( cudaMemcpy( d_image, h_image, IN_BYTES, cudaMemcpyHostToDevice ) );
  checkCUDA( cudaMemcpy( d_filter, h_filter, FILTER_BYTES, cudaMemcpyHostToDevice ) );
  // initialize output to 0, since compute it by adding to it. 
  // TODO: is this better than just having each thread initialize its loc to 0?
  checkCUDA( cudaMemset( d_out, 0, OUT_BYTES ) );

  // setup grid and block sizes
  dim3 threads( THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1 );
  dim3 blocks( OUT_X_SIZE / THREADS_PER_BLOCK_X + 1, 
               OUT_Y_SIZE / THREADS_PER_BLOCK_Y + 1, BATCH_SIZE * FILTER_CHANNELS * IN_CHANNELS);


  // start timer
  checkCUDA( cudaEventRecord( start, 0 ) );

  gpu_conv<<< blocks, threads >>> (d_image, d_filter, d_out);
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


  // cleanup
  cudaFreeHost( h_image );
  cudaFreeHost( h_filter );
  free( h_cpu_out );
  cudaFreeHost( h_gpu_out );

  cudaError_t cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceReset failed!");
      return 1;
  }

  return 0;


}
