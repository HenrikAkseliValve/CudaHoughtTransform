/*
* Provides function to turn RGB image to gray image.
*/
#include<stdint.h>

// Bullshit mangling off.
extern "C"{

// Function takes in RGB color image size of of width
// and heights and outputs gray image to imageout.
// DO NOT USE SAME MEMORY AREA AS INPUT AND OUTPUT!
__global__ void rgbToGray(uint8_t * __restrict__ image,uint8_t * __restrict__ imageout,uint32_t width,uint32_t height){

	// libṕng manual says that RGB to grayscale can be done
	// with followíng formula: Y=(6968 * R + 23434 * G + 2366 * B)/32768.
	// PNG also has cHRM chunk where one can get weights but don't have
	// time to deal with that.
	// http://www.libpng.org/pub/png/libpng-manual.txt

	uint32_t xindex=blockIdx.x*blockDim.x+threadIdx.x;
	uint32_t yindex=blockIdx.y*blockDim.y+threadIdx.y;

	if(xindex<width && yindex<height){
		// Index to the pixel.
		uint32_t index=yindex*width+xindex;
		imageout[index]=(6968*image[index*3+0]+23434*image[index*3+1]+2366*image[index*3+2])/32768;
	}

}

}
