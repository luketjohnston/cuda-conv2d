This project is an ongoing attempt to optimize a CUDA implementation of direct 2d convolution.

All parameters (i.e. image size, filter size, etc) are currently constants in kernel.cu.


when "compare_with_cudnn" is set in kernel.cu, the executable produced by "make" will
run both my implementation, and the cudnn implementation, and print the time each takes.
Currently my implementation is much slower than the cudnn implementation for 
larger inputs (my implementation achieves around 280 Gflops depending on parameters).



TODO:

based on nsight-compute source profiling, the main bottleneck is the loading of the image tiles
into shared memory, and the __syncthreads() afterward. In the current implementation,
each tile of the input image is loaded into shared memory once for each output channel
(since different threads compute different output channels). So I'm currently working
on updating this, so ideally each tile only has to be read from global memory into shared memory
once. 
