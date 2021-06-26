BIN=x.conv
NVCC=nvcc
NVOPTS=-O3 $(ARCH) -DDEBUG -lineinfo -lcudnn
#NVOPTS=-O3 $(ARCH) -lineinfo

$(BIN): 
	$(NVCC) $(NVOPTS) -lcublas kernel.cu cudnn_conv.cu -o $(BIN) 

clean:
	rm -rf kernel.o cudnn_conv.o $(BIN)
