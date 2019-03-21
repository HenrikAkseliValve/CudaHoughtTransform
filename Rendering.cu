/*
* Run simple plotting algorithms.
*/
#include<stdint.h>

// Disable mangling.
extern "C"{

	// Draws multiple lines to output images.
	// Drawing uses polar coordinates.
	// Parameters in are tick angle and radius of the line
	// which satisfy equation:
	//   <radius tick>*<radius diff>=x*cos(<angle tick>*<angle diff>)+y*sin(<angle tick>*<angle diff>)
	__global__ void renderLinesPolar(uint32_t *lineparameters,uint32_t lineparametercount,uint8_t *frame,uint32_t width,uint32_t height,float angled,float radiusd){

    // Calculate and check index in the list for this
    // thread.
    int index=blockIdx.x*gridDim.x+threadIdx.x;
    if(index<lineparametercount){

			float angle=(lineparameters[index]>>16)*angled;
			float radius=((uint16_t)lineparameters[index])*radiusd;
			float sinangle;
			float cosangle;
			sincosf(angle,&sinangle,&cosangle);

			// Since equation to satisfy is
			// <radius>=x*cos(<angle>)+y*sin(<angle>)
			// then
			// y=x*cos(<angle>)*sin(<angle>)^(-1)+<radius>*sin(<angle>)^(-1)
			// if sin(<angle>)!=0.
			// IF sin(<angle>)==0 we have vertical line

			if(sinangle==0){
				// Loop over x dimension (width) and calculate
				// closest y dimension (height) pixel.
				for(uint32_t x=0;x<width;x++){
					// Calculate closes y to used in the frame.
					double yprecise=round((x*cosangle+radius)/sinangle);
					uint32_t y=(uint32_t)(yprecise+1/2);
					if(y<height) frame[y*width+x]=255;
					else break;
				}
			}
			else{
				// Draw horizontal line at radius distance away.
				uint32_t x=round(radius);
				for(uint32_t y=0;y<height;y++) frame[y*width+x]=255;
			}

    }
	}

// End bracket for mangling.
}
