# Compile hought transfrom program

# Compiler used for c files
CC=gcc
# Compilers used for cu files
NVCC:=nvcc --compiler-bindir $(CC)
# Compilation flags
NVCCFLAGS:= -m64 -G
# This is a test program so debug
# For now debug information is forced.
CFLAGS:=-Wall -g

.PHONY: all clean

all: libhough.fatbin hough

libhough.fatbin: RGBtoGray.o Edge.o Hough.o Rendering.o
	$(NVCC) $(NVCCFLAGS) --fatbin --device-link --generate-code=arch=compute_30,code=sm_30 --generate-code=arch=compute_32,code=sm_32 --generate-code=arch=compute_35,code=sm_35 -o $@ $^

RGBtoGray.o: RGBtoGray.cu
	$(NVCC) $(NVCCFLAGS) --device-c -o $@ $^
Edge.o: Edge.cu
	$(NVCC) $(NVCCFLAGS) --device-c -o $@ $^
Hough.o: Hough.cu
	$(NVCC) $(NVCCFLAGS) --device-c -o $@ $^
Rendering.o: Rendering.cu
	$(NVCC) $(NVCCFLAGS) --device-c -o $@ $^

hough: HoughCUDA.c
	$(CC) $(CFLAGS) -o$@ $^ -lcuda -lpng16 -lm
