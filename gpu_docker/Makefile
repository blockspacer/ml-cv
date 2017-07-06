
CUDA=

all:
	nvcc main.cu -o census \
		-Xlinker `pkg-config --cflags --libs opencv`