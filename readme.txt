This project is an ongoing attempt to optimize a CUDA implementation of direct 2d convolution.

All parameters are currently constants in kernel.cu.


when "compare_with_cudnn" is set in kernel.cu, the executable produced by "make" will
run both my implementation, and the cudnn implementation, and print the time each takes.
Currently my implementation is much slower than the cudnn implementation for 
larger inputs. 
