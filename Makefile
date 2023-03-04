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
# What libaries should be added.
LIBS:=-lcuda -lm 

# libpng has to be queried since
# .so file has version number as
# a name postfix.
# If libpng isn't available then 
# NO_LIBPNG has to be set to 1.
NO_LIBPNG?=0
ifeq ($(NO_LIBPNG),1)
  CFLAGS+=-DNO_LIBPNG
else
  LIBS+=-l $(shell ldconfig -p | grep png | head -n1 | cut -d "." -f1 | cut -d "b" -f2)
endif

.PHONY: all clean

all: libhough.fatbin hough

libhough.fatbin: RGBtoGray.o Edge.o Hough.o Rendering.o
	$(NVCC) $(NVCCFLAGS) --fatbin --device-link --generate-code=arch=compute_30,code=sm_30 --generate-code=arch=compute_32,code=sm_32 --generate-code=arch=compute_35,code=sm_35 --generate-code=arch=compute_61,code=sm_61 -o $@ $^

RGBtoGray.o: RGBtoGray.cu
	$(NVCC) $(NVCCFLAGS) --device-c -o $@ $^
Edge.o: Edge.cu
	$(NVCC) $(NVCCFLAGS) --device-c -o $@ $^
Hough.o: Hough.cu
	$(NVCC) $(NVCCFLAGS) --device-c -o $@ $^
Rendering.o: Rendering.cu
	$(NVCC) $(NVCCFLAGS) --device-c -o $@ $^

hough: HoughCUDA.c
	$(CC) $(CFLAGS) -o$@ $^ $(LIBS)
