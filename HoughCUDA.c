/*
* Performs Hoght transdform given image in the
* parameters uing CUDA. To keep things simple
* file includes copy pasted functions for
* handling PNG files.
*/
#include<getopt.h>
#include<unistd.h>
#include<fcntl.h>
#include<string.h>
#include<math.h>
#include<png.h>
#include<cuda.h>
#include"PrintTools.h"


// Simple structure that such has
// information about the PNG image.
// It doesn't store the image data.
//
// Members:
//   width is number of the columns
//   height is number of rows.
//   colortype RGBA or something else.
//   bitdepth is number bits per
//     channel.
//   special is weird one. It
//     stores inside of it result
//     from png_get_valid which
//     answer for example is image
//     sRGB or not.
typedef struct{
	png_uint_32 width;
	png_uint_32 height;
  uint32_t special;
  uint32_t allocationsize;
  int colortype;
	int bitdepth;
	int interlace;
	int compression;
	int filter;
	int rowsize;
}PngInfo;

// Reads PNG image to CUDA memory area to
// given handle to one area with out row
// pointers.
int pngRead(PngInfo *png,CUdeviceptr *gpumemory,FILE *fp){

	// Check the header is correct.
	unsigned char header[8];
	fread(header,1,8,fp);
	if(!png_sig_cmp(header,0,8)){

		png_structp pngstruct = png_create_read_struct(PNG_LIBPNG_VER_STRING,0,0,0);
		if(pngstruct){

			png_infop pnginfo = png_create_info_struct(pngstruct);
			if(pnginfo){

				// Do I really need to setjmp.
				// Yes and no. libpng has to be compiled with with PNG_NO_SETJMP to not have to do this.
				if(setjmp(png_jmpbuf(pngstruct))){
					(void)write(STDERR_FILENO,"ERROR: With in the libpng\n",26);
					// This command destroyies both pngstruct and pnginfo.
					png_destroy_read_struct(&pngstruct,&pnginfo,0);
					return 0;
				}

				// Default behavior.
				// You could replace final writting, reading, etc functions but why would you?
				png_init_io(pngstruct,fp);

				// This kind awful design since png_sig_cmp can take 1 to 8 bytes in it
				// At this point programmer has to link amount already read with the png_struct
				// so that library reads correct places.
				png_set_sig_bytes(pngstruct,8);

				// TODO: How to handle paletted images since bit per channel isn't constant.
				// TODO: Does interlacing need special handling?
				png_read_info(pngstruct,pnginfo);
				png_get_IHDR(pngstruct,pnginfo,&png->width,&png->height,&png->bitdepth,&png->colortype,&png->interlace,&png->compression,&png->filter);
				//png->special=png_get_valid(pngstruct,pnginfo,PNG_INFO_sBIT);

				png->rowsize = png_get_rowbytes(pngstruct,pnginfo);
				png->allocationsize=png->rowsize*png->height;

				// Allocate memory for the image.
				if(cuMemAlloc(gpumemory,png->allocationsize)==CUDA_SUCCESS){

					uint8_t *memory=malloc(png->allocationsize);
					if(memory){

						// New jump point for png_read_row error so that
						// we can free memory allocations.
						if(setjmp(png_jmpbuf(pngstruct))) goto _jmp_ERROR_EXIT;

						// Reads rows to one memory area.
						for(uint32_t i=0;i<png->height;i++) png_read_row(pngstruct,memory+i*png->rowsize,0);

						// Move to device
						if(cuMemcpyHtoD(*gpumemory,memory,png->allocationsize)==CUDA_SUCCESS){

							free(memory);
							// Free allocated resources.
							png_destroy_read_struct(&pngstruct,&pnginfo,0);
							return 1;
						}

						// Jump here happens if long jump from
						_jmp_ERROR_EXIT:
						free(memory);
					}
					else (void)write(STDERR_FILENO,"ERROR: pngRead | malloc\n",26);
					cuMemFree(*gpumemory);
				}
				else (void)write(STDERR_FILENO,"ERROR: pngRead | cuMemAlloc\n",28);

				// Destyroy info structure now so that next we can destroy
				// the png_struct.
				png_destroy_info_struct(pngstruct,&pnginfo);

			}
			// Only free the png_struct since png_info couldn't be created.
			png_destroy_read_struct(&pngstruct,0,0);
		}
		// png_struct couldn't be created!
	}
	// File given wasn't PNG file!

	return 0;
}

// Write CUDA device memory to png file.
int pngWrite(const PngInfo *info,const CUdeviceptr gpumemory,FILE *fp){

  png_structp pngstruct=png_create_write_struct(PNG_LIBPNG_VER_STRING,0,0,0);
  if(pngstruct){
		png_infop pnginfo=png_create_info_struct(pngstruct);
		if(pnginfo){
			// Allocate temporary memory region of memory.
			png_bytep memory=malloc(sizeof(png_byte)*info->allocationsize);
			if(memory){

				// Move the gpumemory
				cuMemcpyDtoH(memory,gpumemory,sizeof(png_byte)*info->allocationsize);

        // Setup longjump for libpng to return to
        // if it encounters an error.
				if(setjmp(png_jmpbuf(pngstruct))){
					(void)write(STDERR_FILENO,"ERROR: With in the libpng\n",26);
					free(memory);
					png_destroy_write_struct(&pngstruct,&pnginfo);
					return 0;
				}

				// Default underline behavior.
				png_init_io(pngstruct,fp);

				// Write header.
				// Parameters are:
				// Width,height,bit depth,color type,interlace type,compression type,filter type
				png_set_IHDR(pngstruct,pnginfo,info->width,info->height,info->bitdepth,info->colortype,info->interlace,info->compression,info->filter);

				png_write_info(pngstruct,pnginfo);

        for(uint32_t i=0;i<info->height;i++) png_write_row(pngstruct,memory+i*info->rowsize);

        png_write_end(pngstruct,pnginfo);


				png_free_data(pngstruct,pnginfo,PNG_FREE_ALL,-1);
				png_destroy_write_struct(&pngstruct,&pnginfo);
				free(memory);

				return 1;
			}
			else (void)write(STDERR_FILENO,"ERROR: pngWrite | malloc\n",33);
			png_destroy_info_struct(pngstruct,&pnginfo);
		}
		png_destroy_write_struct(&pngstruct,0);
  }

	return 0;
}
// Main entry to the program.
// Does initialiation and GPU ordering.
//
// Commandline arguments:
//  Non-options are images to be hough
//    tranformed. For now have to PNGs.
//  -g index selecs the GPU of given index.
//  -e threshold selects threshold used in
//    edge detection.
//
int main(int argn,char **args){

	// Initialization function for CUDA.
  // Flag is zero since it has to be!
	if(cuInit(0)==CUDA_SUCCESS){

		// Which GPU to use. Defaults to first one.
		int selectedgpu=0;
		// Númber of GPU in the system.
		int numberofgpus=0;
		// Edge detection threshold
		float edgethreshold=0.2;
		if(cuDeviceGetCount(&numberofgpus)==CUDA_SUCCESS){

			// Handle Arguments with getopt.
			// For simplicity don't use long options.
			{
				int c;
				while((c=getopt(argn,args,"g:"))!=-1){
					switch(c){
						case 'g':
							selectedgpu=atoi(optarg);
							break;
						case 'e':
							edgethreshold=atof(optarg);
							break;
					}
				}
			}

			// Get handler for GPU to be used
			// for this hought transform. Make
			// sure that selectedgpu is less
			// the numberofgpus.
      if(selectedgpu<numberofgpus){
				CUdevice gpu;
				if(cuDeviceGet(&gpu,selectedgpu)==CUDA_SUCCESS){

					// Maximum number threads.
					int maxthreads;
					cuDeviceGetAttribute(&maxthreads,CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,gpu);
					// Maximum number of threads in x dimension.
					int maxblockdimx;
					cuDeviceGetAttribute(&maxblockdimx,CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,gpu);
					// Maximum number of threads in y dimension.
					int maxblockdimy;
					cuDeviceGetAttribute(&maxblockdimy,CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,gpu);
					// Maximum number of blocks in x dimemsion.
					int maxgriddimx;
					cuDeviceGetAttribute(&maxgriddimx,CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,gpu);
					// Maximum number of blocks in y dimemsion.
					int maxgriddimy;
					cuDeviceGetAttribute(&maxgriddimy,CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,gpu);
					// Maximum amount of shared memory per block in bytes.
					int maxshared;
					cuDeviceGetAttribute(&maxshared,CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,gpu);
					// Amount of threads in warp (number of threads executed simultaneously).
					int warpsize;
					cuDeviceGetAttribute(&warpsize,CU_DEVICE_ATTRIBUTE_WARP_SIZE,gpu);
          // Number of processor
          int gpuprocessors;
          cuDeviceGetAttribute(&gpuprocessors,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,gpu);

					// Create context for GPU cpu interaction.
					CUcontext context;
					if(cuCtxCreate(&context,CU_CTX_SCHED_SPIN,gpu)==CUDA_SUCCESS){

						// Load the "shared library" which has GPU code and
						// "get address" to functions.
						CUmodule libhough;
						if(cuModuleLoad(&libhough,"libhough.fatbin")==CUDA_SUCCESS){
							CUfunction rgbtograykernel;
							if(cuModuleGetFunction(&rgbtograykernel,libhough,"rgbToGray")==CUDA_SUCCESS){
								CUfunction sobelkernel;
								if(cuModuleGetFunction(&sobelkernel,libhough,"sobel")==CUDA_SUCCESS){
									CUfunction houghlinekernel;
									if(cuModuleGetFunction(&houghlinekernel,libhough,"houghLine")==CUDA_SUCCESS){
										CUfunction renderlineskernel;
										if(cuModuleGetFunction(&renderlineskernel,libhough,"renderLinesPolar")==CUDA_SUCCESS){

											// LOAD THE IMAGE LOOP
											// Loop for over every PNG image.
											// Loop goes through all the
											for(char *file=args[optind];optind<argn;file=args[++optind]){
												FILE *readfd=fopen(file,"rb");
												if(readfd){
													PngInfo pnginfo;
													CUdeviceptr image;
													if(pngRead(&pnginfo,&image,readfd)){

														// CALCULATE GRID AND BLOCK SIZE
														// TODO: What happens if image is too small.
														// TODO: What if image is so big there
														//       need to have loop inside of the GPU code?
														unsigned int blockx=warpsize;
														unsigned int blocky=warpsize;

														unsigned int gridx=pnginfo.width/blockx+(pnginfo.width/blockx>0);
														unsigned int gridy=pnginfo.height/blocky+(pnginfo.height/blocky>0);


														// Change color images to gray scale and continue with gray scale images.
														CUdeviceptr grayimage;
														switch(pnginfo.colortype){
															case PNG_COLOR_TYPE_RGB:
																if(cuMemAlloc(&grayimage,pnginfo.width*pnginfo.height)==CUDA_SUCCESS){

																	// Execute RGB to Gray kernel to get gray.
																	void *args[]={&image,&grayimage,&pnginfo.width,&pnginfo.height,0};

																	if(cuLaunchKernel(rgbtograykernel,gridx,gridy,1,blockx,blocky,1,0,0,args,0)!=CUDA_SUCCESS){
																		(void)write(STDERR_FILENO,"ERROR: cuLaunchKernel | grayimage\n",35);
																		goto jmp_SAFE_EXIT_GRAYIMAGE;
																	}

																}
																else{
																	(void)write(STDERR_FILENO,"ERROR: cuMemAlloc | grayimage\n",30);
																	goto jmp_SAFE_EXIT_GRAYIMAGE;
																}
																break;
															case PNG_COLOR_TYPE_GRAY:
																grayimage=image;
																break;
															default:
																fprintf(stderr,"\nSkipping the file \"%s\" since it has unimplemented colortype!\n",file);
																cuMemFree(image);
																continue;
														}

														// Run Egde detection. Output will be list of indexes (y*width+x)
														// and number memory allocated for that list is used.
														CUdeviceptr binedge;
														if(cuMemAlloc(&binedge,pnginfo.width*pnginfo.height*sizeof(uint8_t))==CUDA_SUCCESS){
															CUresult result;
															{
																void *args[]={&grayimage,&binedge,&pnginfo.width,&pnginfo.height,&edgethreshold,0};
																result=cuLaunchKernel(sobelkernel,gridx,gridy,1,blockx,blocky,1,0,0,args,0);
															}
															if(result==CUDA_SUCCESS){

																// Make edge list from grayimage manually on CPU since we have to
																// count up which isn't easy to do.
																// TODO: Design how to do this on GPU at edge detector!
																uint32_t *deviceedgelist=malloc(sizeof(uint32_t)*pnginfo.width*pnginfo.height);
																if(deviceedgelist){
																	uint32_t edgelistcount=0;
																	uint8_t *edgeimagedevice=malloc(sizeof(uint8_t)*pnginfo.width*pnginfo.height);
																	if(edgeimagedevice){
																		cuMemcpyDtoH(edgeimagedevice,binedge,sizeof(uint8_t)*pnginfo.width*pnginfo.height);
																		// Don't need edgeimage anymore.
																		cuMemFree(binedge);

																		for(uint32_t x=0;x<pnginfo.width;x++){
																			for(uint32_t y=0;y<pnginfo.height;y++){
																				if(edgeimagedevice[y*pnginfo.width+x]>0) deviceedgelist[edgelistcount++]=(x<<16)+y;
																			}
																		}

																		// If we don't have enough edge pixels, then
																		// there isn't much of point continuing.
																		// Warpsize used as counter so that later
																		// when calculating kernel grid sizes isn't
																		// zero.
																		// TODO: Better grid size calculation on hough transform.
																		if(edgelistcount>warpsize*2){

																			CUdeviceptr edgelist;
																			if(cuMemAlloc(&edgelist,sizeof(uint32_t)*edgelistcount)==CUDA_SUCCESS){
																				free(deviceedgelist);
																				free(edgeimagedevice);

																				// Run the hough transform.

																				// Calculate angle and distance difference used in accumulator.
																				// Angle should be between 0 and <PI>/2 and radius should be
																				// between zero and image diagonal (sqrt(width^2+height^2)).
																				// Calculate number of ticks based upon this number which
																				// are floored because extreame values probably don't exist.
																				float maxradius=sqrt(pnginfo.width*pnginfo.width+pnginfo.height*pnginfo.height);
																				float angled=0.001;
																				float radiusd=2;
																				uint16_t angleticks=(uint16_t)floor((M_PI/2)/angled);
																				uint16_t radiusticks=(uint16_t)floor(maxradius/radiusd);

																				// Allocate accumator based upon number ticks we have.
																				// Also memset to zero so that we have clean memory.
																				CUdeviceptr accumulator;
																				if(cuMemAlloc(&accumulator,sizeof(uint32_t)*angleticks*radiusticks)==CUDA_SUCCESS){
																					cuMemsetD32(accumulator,0,angleticks*radiusticks);
																					// Run thread per edge index.
																					// Use only dimension as input is index list.
																					{
																						void *args[]={&edgelist,&edgelistcount,&accumulator,&angled,&radiusd,&angleticks,&radiusticks,0};
																						result=cuLaunchKernel(houghlinekernel,(edgelistcount/warpsize)+(edgelistcount%warpsize>0),1,1,warpsize,1,1,0,0,args,0);
																					}
																					if(result==CUDA_SUCCESS){

																						// Collect information for render the lines.
																						uint32_t *accumulatorhost=malloc(sizeof(uint32_t)*angleticks*radiusticks);
																						if(accumulatorhost){
																							cuMemcpyDtoH(accumulatorhost,accumulator,sizeof(uint32_t)*angleticks*radiusticks);
																							uint32_t *hostlineparameters=malloc(sizeof(uint32_t)*edgelistcount);
																							if(hostlineparameters){

																								uint32_t peakthreas=2800;
																								uint32_t peakcount=0;
																								for(uint16_t angle=0;angle<angleticks;angle++){
																									for(uint16_t radius=0;radius<radiusticks;radius++){
																										if(accumulatorhost[radius*angleticks+angle]>peakthreas){

																											hostlineparameters[peakcount++]=(angle<<16)+radius;

																										}
																									}
																								}

																								// Allocate memory for making lines images.
																								CUdeviceptr lineparameters;
																								if(cuMemAlloc(&lineparameters,sizeof(uint32_t)*peakcount)==CUDA_SUCCESS){
																									cuMemcpyHtoD(lineparameters,hostlineparameters,sizeof(uint32_t)*peakcount);
																									CUdeviceptr finalimage;
																									if(cuMemAlloc(&finalimage,pnginfo.width*pnginfo.height)==CUDA_SUCCESS){
																										cuMemsetD8(finalimage,0,pnginfo.width*pnginfo.height);
																										// Give rendering information.
																										{
																											void *args[]={&lineparameters,&peakcount,&finalimage,&pnginfo.width,&pnginfo.height,&angled,&radiusd};
																											result=cuLaunchKernel(renderlineskernel,peakcount/warpsize+(peakcount%warpsize>0),1,1,warpsize,1,1,0,0,args,0);
																										}
																										if(result==CUDA_SUCCESS){

																											// Make sure image to be written out is gray image with 8 bit channel.
																											pnginfo.allocationsize=pnginfo.width*pnginfo.height;
																											pnginfo.rowsize=pnginfo.width;
																											pnginfo.colortype=PNG_COLOR_TYPE_GRAY;
																											pnginfo.bitdepth=8;

																											FILE *wfd=fopen("test.png","wb");
																											pngWrite(&pnginfo,accumulator,wfd);
																											fclose(wfd);

																										}
																										else (void)write(STDOUT_FILENO,"ERROR: cuLaunchKernel | renderLines\n",36);
																										cuMemFree(finalimage);
																									}
																									else (void)write(STDOUT_FILENO,"ERROR: cuMemAlloc | finalimage\n",33);
																									cuMemFree(lineparameters);
																								}
																								else (void)write(STDOUT_FILENO,"ERROR: cuMemAlloc | lineparameters\n",37);
																								free(hostlineparameters);
																							}
																							else (void)write(STDOUT_FILENO,"ERROR: malloc failled!\n",23);
																							free(accumulatorhost);
																						}
																						else (void)write(STDOUT_FILENO,"ERROR: malloc failled!\n",23);
																					}
																					else (void)write(STDOUT_FILENO,"ERROR: cuLaunchKernel | hough\n",30);
																				}
																			}
																		}
																		else{
																			free(deviceedgelist);
																			free(edgeimagedevice);
																			(void)write(STDOUT_FILENO,"No edge pixel found\n",20);
																		}
																	}
																	else{
																		free(deviceedgelist);
																		(void)write(STDOUT_FILENO,"ERROR: malloc failled!\n",23);
																	}
																}
																else (void)write(STDOUT_FILENO,"ERROR: malloc failled!\n",23);

															}
															else{
																cuMemFree(binedge);
																(void)write(STDERR_FILENO,"ERROR: cuLaunchKernel | sobelkernel\n",36);
															}
														}
														else (void)write(STDERR_FILENO,"ERROR: cuMemAlloc | binedge\n",28);

														// Program jumps here if rgbToGray
														// errors for some reason.
														jmp_SAFE_EXIT_GRAYIMAGE:
														// If image was already gray we don't
														// need to free gray version of it.
														if(grayimage!=image) cuMemFree(grayimage);

														cuMemFree(image);
													}
													else fprintf(stderr,"\nPNG read error happened to \"%s\".\nProgram continues despite this!\n",file);
													fclose(readfd);
												}
												else fprintf(stderr,"\nPNG read error happened to \"%s\"!\n      Program continues despite this!\n",file);
											}
										}
										else (void)write(STDERR_FILENO,"ERROR: Kernel | renderLines\n",28);
									}
									else (void)write(STDERR_FILENO,"ERROR: Kernel | houghline\n",26);
								}
								else (void)write(STDERR_FILENO,"ERROR: Kernel | sobel\n",22);
							}
							else (void)write(STDERR_FILENO,"ERROR: Kernel | rgbToGray\n",26);

							// Unload the module
							cuModuleUnload(libhough);
						}
						else (void)write(STDERR_FILENO,"ERROR: cuModuleLoad | libhough\n",31);

						// Since loop is behind us just destroy the GPU context.
						cuCtxDestroy(context);
					}
					else write(STDERR_FILENO,"ERROR: cuCtxCreate\n",19);
				}
				else write(STDERR_FILENO,"ERROR: cuDeviceGet\n",20);
      }
      else write(STDERR_FILENO,"ERROR: selectedgpu is more of equal to number GPUs\n",54);
		}
		else write(STDERR_FILENO,"ERROR: cuDeviceGetCount\n",25);
	}
	else write(STDERR_FILENO,"ERROR: cuInit\n",14);

	return 0;
}
