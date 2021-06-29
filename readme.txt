This project is an ongoing attempt to optimize a CUDA implementation of direct 2d convolution.

All parameters (i.e. image size, filter size, etc) are currently constants in kernel.cu.

when "compare_with_cudnn" is set in kernel.cu, the executable produced by "make" will
run both my implementation, and the cudnn implementation, and print the time each takes.
Currently my implementation is much slower than the cudnn implementation for 
larger inputs (my implementation achieves a max of around 525 Gflops depending on parameters and run).

when "compare_with_cpu" is set, the executable will run a very naive (non optimized) CPU implementation 
and compare with my GPU implementation.


TODOS:
According to the nsight-compute source profiling, the
bottleneck currently seems be shared memory access in order to accumulate "acc" by multiplying filter and window together.

if I just remove the __syncthreads(), somehow the kernel slows down. Maybe it makes global memory accesses
less regular so there's less cache hits? Worth investigating... In any case that doesn't seem to be the primary bottleneck.

Probably need to utilize the tensor cores to achieve close to cudnn performance.

I replaced the "int" types with "long" (typedef long intType_t), which allows working with some larger inputs and filters,
but this cost some performance. Ideally, should probably automatically split up very large inputs into multiple kernel calls.


COMPLETED TODOS:
based on nsight-compute source profiling, the main bottleneck is the loading of the image tiles
into shared memory, and the __syncthreads() afterward. In the current implementation,
each tile of the input image is loaded into shared memory once for each output channel
(since different threads compute different output channels). So I'm currently working
on updating this, so ideally each tile only has to be read from global memory into shared memory
once. 
