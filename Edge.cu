/*
* Gives functions for egde detection.
*/
#include<stdint.h>
#include<math.h>

// Mangling disabling.
extern "C"{

// Sobel takes derivative as convolution.
// 3 by 3 mask is horizontially is
//  _        _
// |  -1 0 1  |
// |  -2 0 2  |
// |_ -1 0 1 _|
//
// and vertically
//  _          _
// |  -1 -2 -1  |
// |   0  0  0  |
// |_  1  2  1 _|
//
// Source: http://www.aishack.in/tutorials/sobel-laplacian-edge-detectors/
__global__ void sobel(uint8_t * __restrict__ grayimage,uint8_t * __restrict__ binaryout,uint32_t width,uint32_t height,float threshold){

  uint32_t xindex=blockIdx.x*blockDim.x+threadIdx.x;
  uint32_t yindex=blockIdx.y*blockDim.y+threadIdx.y;

  // Check that we aren't at the border pixel or outside of the image.
  if(0<xindex && xindex<width-1 && 0<yindex && yindex<height-1){

    // Horizontial mask
    int16_t dx=0;
    dx-=grayimage[(yindex-1)*width+xindex-1];
		dx-=2*grayimage[yindex*width+xindex-1];
		dx-=grayimage[(yindex+1)*width+xindex-1];
		dx+=grayimage[(yindex-1)*width+xindex+1];
		dx+=2*grayimage[yindex*width+xindex+1];
		dx+=grayimage[(yindex+1)*width+xindex+1];

		// Vertical mask
		int16_t dy=0;
    dy-=grayimage[(yindex-1)*width+xindex-1];
		dy-=2*grayimage[(yindex-1)*width+xindex];
		dy-=grayimage[(yindex-1)*width+xindex+1];
		dy+=grayimage[(yindex+1)*width+xindex-1];
		dy+=2*grayimage[(yindex+1)*width+xindex];
		dy+=grayimage[(yindex+1)*width+xindex+1];

    // Approximate strength is abs(dx)+abs(dy)
    // which at biggest is 4*255+4*255=2040 so normalize
    // threshold to it.
    // TODO: Calculate normalization of threshold!
    binaryout[yindex*width+xindex]=(abs(dx)+abs(dy)>2040*threshold);
  }
  else if(xindex==0 || yindex==0 || xindex==width || yindex==height)  binaryout[yindex*width+xindex]=0;
}

// Bullhock extern "C" end...
// why can't there be option
// to disable mangling...
}
