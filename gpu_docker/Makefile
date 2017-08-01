
#Please make sure that
# -   PATH includes /usr/local/cuda-8.0/bin
# -   LD_LIBRARY_PATH includes /usr/local/cuda-8.0/lib64, or, add /usr/local/cuda-8.0/lib64 to /etc/ld.so.conf and run ldconfig as root


CUDA=
# PATH=$(PATH):/usr/local/cuda-8.0/bin
NVCC=/usr/local/cuda-8.0/bin/nvcc
#nvcc

# http://www.linuxquestions.org/questions/programming-9/can%27t-execute-c-binaries-permission-denied-even-though-permission-is-777-a-102959/
# fixme: ./a.out bash: ./a.out: Permission denied

#/usr/local/cuda-8.0/bin/nvcc
LIBS=`pkg-config --cflags --libs opencv ` -std=c++11

all:
	$(NVCC) main.cu -o census $(LIBS)
	
	# doesn't work
	#$(NVCC) main.cu -o census -Xlinker $(LIBS)
	# -Xlinker `pkg-config --cflags --libs opencv`