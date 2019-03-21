/*
* Calculates Hough transform from
* given edge binary image.
*/
#include<stdint.h>
#include<math.h>

// Bulhock ectern "C" to disable
// mangling.
extern "C"{

// Hough transform from edge list for line fit.
// Line equation is <radius>=x*cos(<angle>)+y*sin(<angle>)
// where radius and theta are dimensions of the
// accumulator.
// x and y are stored in on 32 bit interger.
__global__ void houghLine(uint32_t *edgelist,uint32_t edgelistcount,uint32_t *accumulator,float angled,float radiusd,uint16_t angleticks,uint16_t radiusticks){


	// Index for this thread in edgelist.
	uint32_t index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<edgelistcount){

		// Get the x and y coordinates of the edge.
		uint32_t x=edgelist[index]>>16;
		uint16_t y=(uint16_t)edgelist[index];

		// Go through the angles and calculate real
		// radius (<radius>=x*cos(<theta>)+y*sin(<theta>))
		// and then calculate nearest radius pixel from
		// accumulator.
		for(uint32_t angle=0;angle<angleticks;angle++){
			float cosangle;
			float sinangle;

			// Offset the loop by index do prevent later
			// atomicAdd to happen same location and
			// such hope that memory conficts are resolved.
			uint32_t angleoff=(angle+index)%angleticks;

			// Calculate sin and cosine of the angle.
			sincosf(angleoff*angled,&sinangle,&cosangle);
			double realradius=x*cosangle+y*sinangle;

			// We need to find closest radius pixel.
			// Since we should not use for to search
			// since that would cause warp to be unsynced
			// we have to use flooring.
			// Following equation holds:
			//  round(x)=floor(x+1/2)
			// We can generalize and say so:
			//	x=floor(x/radiusd+1/2);
			uint32_t radiusindex=(uint32_t)(realradius/radiusd+1/2);

      atomicAdd(accumulator+(radiusindex*angleticks+angleoff),1);

		}

	}
}

// Extern "C" end.
}
