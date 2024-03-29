/*
* Performs Hought transform given image in the
* parameters uing CUDA. To keep things simple
* file includes copy pasted functions for
* handling PNG files.
*/
#include <getopt.h>
#include <unistd.h>
#include <sys/uio.h>
#include <fcntl.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <cuda.h>
#ifndef NO_LIBPNG
	#include <png.h>
#endif /* NO_LIBPNG */

// Write constant message to the console.
#define msg(f,s) (void)write(f,s "\n",sizeof(s))
// Write constant error message to the console.
#define err(s) msg(STDERR_FILENO,"ERROR: " s)
// Write error message and CUDA error code to the console.
#define errcuda(s,e) {\
	err(s); \
	const char *str; \
	cuGetErrorName(e,&str); \
	struct iovec vecs[2]={{.iov_base=(char*)str,.iov_len=strlen(str)},{.iov_base="\n",.iov_len=1}}; \
	(void)writev(STDERR_FILENO,vecs,2); \
}

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
	
	#ifdef NO_LIBPNG
	// Since we don't have libpng we have to improvise
	
	#else
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
					err("With in the libpng!");
					// This command destroys both pngstruct and pnginfo.
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
				// If image is RGB allocation times 3.
				if(png->colortype==PNG_COLOR_TYPE_RGB) png->allocationsize*=3;

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

						// Jump here happens if long jump from PNG error.
						_jmp_ERROR_EXIT:
						free(memory);
					}
					else err("pngRead | malloc!");
					cuMemFree(*gpumemory);
				}
				else err("pngRead | cuMemAlloc!");

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
	#endif /* NO_LIBPNG */
}
// Write host memory image to PNG file.
int pngWriteHost(const PngInfo *info,const png_bytep memory,FILE *fp){
	
	png_structp pngstruct=png_create_write_struct(PNG_LIBPNG_VER_STRING,0,0,0);
	if(!pngstruct) return 0;
	
	png_infop pnginfo=png_create_info_struct(pngstruct);
	if(!pnginfo){
		png_destroy_write_struct(&pngstruct,0);
		return 0;
	}
		
	// Setup longjump for libpng to return to
	// if it encounters an error.
	if(setjmp(png_jmpbuf(pngstruct))){
		err("With in the libpng!");
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

	// Free Libpng write resources.
	png_free_data(pngstruct,pnginfo,PNG_FREE_ALL,-1);
	png_destroy_info_struct(pngstruct,&pnginfo);
	png_destroy_write_struct(&pngstruct,0);
	
	return 1;
}

// Write CUDA device memory to PNG file.
int pngWrite(const PngInfo *info,const CUdeviceptr gpumemory,FILE *fp){

	#ifdef NO_LIBPNG
	// No libpng available has to improvise.
	
	#else
	
	// Allocate temporary memory region of memory.
	png_bytep memory=malloc(sizeof(png_byte)*info->allocationsize);
	if(memory){

		// Move the gpumemory
		cuMemcpyDtoH(memory,gpumemory,sizeof(png_byte)*info->allocationsize);

		int result=pngWriteHost(info,memory,fp);
		
		free(memory);
		
		return result;
	}
	else err("pngWrite | malloc!");
	return 0;
	

	#endif /* NO_LIBPNG */
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

	// Do get more knowledgable abouts errors have general
	// CUResult capture.
	CUresult ecode;

	// Debugging flags.
	struct{
		bool gray:1;
		bool edge:1;
	}debugflags={false,false};

	// Initialization function for CUDA.
	// Flag is zero since it has to be!
	if((ecode=cuInit(0))==CUDA_SUCCESS){

		// Which GPU to use. Defaults to first one.
		int selectedgpu=0;
		// N�mber of GPU in the system.
		int numberofgpus=0;
		// Edge detection threshold
		float edgethreshold=0.2;
		if((ecode=cuDeviceGetCount(&numberofgpus))==CUDA_SUCCESS){

			// Handle Arguments with getopt.
			// For simplicity don't use long options.
			{
				int c;
				while((c=getopt(argn,args,"hg:e:d:"))!=-1){
					switch(c){
						case 'h':
							msg(STDOUT_FILENO,"Usage: hough [options] <PNG file>\n\n"
							                  "Options:\n"
							                  "\t-g <integer> select GPU.\n"
							                  "\t-e <float> give edge threshold (between 0 and 1).\n"
							                  "\t-d <value> debug output.Possible values:\n"
							                  "\t\tgray\twrite colour to gray conversion image.\n"
							                  "\t\tedge\twrite edge detection image."
							);
							break;
						case 'g':
							selectedgpu=atoi(optarg);
							break;
						case 'e':
							edgethreshold=atof(optarg);
							break;
						case 'd':
							if(strcmp(optarg,"gray")==0){
								debugflags.gray=true;
							}
							else if(strcmp(optarg,"edge")==0){
								debugflags.edge=true;
							}
							else{
								err("Unknown value for debugging");
								return 0;
							}
					}
				}
			}

			// Get handler for GPU to be used
			// for this hought transform. Make
			// sure that selectedgpu is less
			// the numberofgpus.
			if(selectedgpu<numberofgpus){
				CUdevice gpu;
				if((ecode=cuDeviceGet(&gpu,selectedgpu))==CUDA_SUCCESS){

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
						if((ecode=cuModuleLoad(&libhough,"libhough.cubin"))==CUDA_SUCCESS){
							CUfunction rgbtograykernel;
							if((ecode=cuModuleGetFunction(&rgbtograykernel,libhough,"rgbToGray"))==CUDA_SUCCESS){
								CUfunction sobelkernel;
								if((ecode=cuModuleGetFunction(&sobelkernel,libhough,"sobel"))==CUDA_SUCCESS){
									CUfunction houghlinekernel;
									if((ecode=cuModuleGetFunction(&houghlinekernel,libhough,"houghLine"))==CUDA_SUCCESS){
										CUfunction renderlineskernel;
										if((ecode=cuModuleGetFunction(&renderlineskernel,libhough,"renderLinesPolar"))==CUDA_SUCCESS){

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
																if((ecode=cuMemAlloc(&grayimage,pnginfo.width*pnginfo.height))==CUDA_SUCCESS){

																	// Execute RGB to Gray kernel to get gray.
																	void *args[]={&image,&grayimage,&pnginfo.width,&pnginfo.height,0};

																	if((ecode=cuLaunchKernel(rgbtograykernel,gridx,gridy,1,blockx,blocky,1,0,0,args,0))!=CUDA_SUCCESS){
																		errcuda("cuLaunchKernel | grayimage!",ecode);
																		goto jmp_SAFE_EXIT_GRAYIMAGE;
																	}
																	
																	// Do user want to write gray image as debug.
																	if(debugflags.gray){
																		FILE *fp=fopen("/tmp/gray.png","wb");
																		PngInfo debugpnginfo;
																		debugpnginfo.width=pnginfo.width;
																		debugpnginfo.height=pnginfo.height;
																		debugpnginfo.special=pnginfo.special;
																		debugpnginfo.allocationsize=pnginfo.width*pnginfo.height;
																		debugpnginfo.colortype=PNG_COLOR_TYPE_GRAY;
																		debugpnginfo.bitdepth=8;
																		debugpnginfo.interlace=0;
																		debugpnginfo.compression=0;
																		debugpnginfo.filter=0;
																		debugpnginfo.rowsize=pnginfo.width;
																		pngWrite(&debugpnginfo,grayimage,fp);
																		fclose(fp);
																	}

																}
																else{
																	errcuda("cuMemAlloc | grayimage!",ecode);
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
														if((ecode=cuMemAlloc(&binedge,pnginfo.width*pnginfo.height*sizeof(uint8_t)))==CUDA_SUCCESS){
															{
																void *args[]={&grayimage,&binedge,&pnginfo.width,&pnginfo.height,&edgethreshold,0};
																ecode=cuLaunchKernel(sobelkernel,gridx,gridy,1,blockx,blocky,1,0,0,args,0);
															}
															if(ecode==CUDA_SUCCESS){

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
																		
																		// If debugging the edge image is enabled write the edge image.
																		if(debugflags.edge){
																			FILE *fp=fopen("/tmp/edge.png","wb");
																			PngInfo debugpnginfo;
																			debugpnginfo.width=pnginfo.width;
																			debugpnginfo.height=pnginfo.height;
																			debugpnginfo.special=pnginfo.special;
																			debugpnginfo.allocationsize=0;
																			debugpnginfo.colortype=PNG_COLOR_TYPE_GRAY;
																			debugpnginfo.bitdepth=8;
																			debugpnginfo.interlace=0;
																			debugpnginfo.compression=0;
																			debugpnginfo.filter=0;
																			debugpnginfo.rowsize=pnginfo.width;
																			pngWriteHost(&debugpnginfo,(png_bytep)edgeimagedevice,fp);
																			fclose(fp);
																		}

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
																			if((ecode=cuMemAlloc(&edgelist,sizeof(uint32_t)*edgelistcount))==CUDA_SUCCESS){
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
																						ecode=cuLaunchKernel(houghlinekernel,(edgelistcount/warpsize)+(edgelistcount%warpsize>0),1,1,warpsize,1,1,0,0,args,0);
																					}
																					if(ecode==CUDA_SUCCESS){

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
																								if(peakcount>0){
																									if((ecode=cuMemAlloc(&lineparameters,sizeof(uint32_t)*peakcount))==CUDA_SUCCESS){
																										cuMemcpyHtoD(lineparameters,hostlineparameters,sizeof(uint32_t)*peakcount);
																										CUdeviceptr finalimage;
																										if((ecode=cuMemAlloc(&finalimage,pnginfo.width*pnginfo.height))==CUDA_SUCCESS){
																											cuMemsetD8(finalimage,0,pnginfo.width*pnginfo.height);
																											// Give rendering information.
																											{
																												void *args[]={&lineparameters,&peakcount,&finalimage,&pnginfo.width,&pnginfo.height,&angled,&radiusd};
																												ecode=cuLaunchKernel(renderlineskernel,peakcount/warpsize+(peakcount%warpsize>0),1,1,warpsize,1,1,0,0,args,0);
																											}
																											if(ecode==CUDA_SUCCESS){

																												// Make sure image to be written out is gray image with 8 bit channel.
																												pnginfo.allocationsize=pnginfo.width*pnginfo.height;
																												pnginfo.rowsize=pnginfo.width;
																												pnginfo.colortype=PNG_COLOR_TYPE_GRAY;
																												pnginfo.bitdepth=8;

																												FILE *wfd=fopen("test.png","wb");
																												pngWrite(&pnginfo,accumulator,wfd);
																												fclose(wfd);

																											}
																											else errcuda("cuLaunchKernel | renderLines!",ecode);
																											cuMemFree(finalimage);
																										}
																										else errcuda("cuMemAlloc | finalimage!",ecode);
																										cuMemFree(lineparameters);
																									}
																									else errcuda("cuMemAlloc | lineparameters!",ecode);
																									free(hostlineparameters);
																								}
																								else msg(STDOUT_FILENO,"No edges found?");
																							}
																							else err("malloc failled!");
																							free(accumulatorhost);
																						}
																						else err("malloc failled!");
																					}
																					else errcuda("cuLaunchKernel | hough!",ecode);
																				}
																			}
																			else errcuda("cuMemAlloc | edge allocation!",ecode);
																		}
																		else{
																			free(deviceedgelist);
																			free(edgeimagedevice);
																			err("No edge pixel found!");
																		}
																	}
																	else{
																		free(deviceedgelist);
																		err("malloc failled!");
																	}
																}
																else err("malloc failled!");

															}
															else{
																cuMemFree(binedge);
																errcuda("cuLaunchKernel | sobelkernel!",ecode);
															}
														}
														else errcuda("cuMemAlloc | binedge!",ecode);

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
										else errcuda("Kernel | renderLines!",ecode);
									}
									else errcuda("Kernel | houghline!",ecode);
								}
								else errcuda("Kernel | sobel!",ecode);
							}
							else errcuda("Kernel | rgbToGray!",ecode);

							// Unload the module
							cuModuleUnload(libhough);
						}
						else errcuda("cuModuleLoad | libhough!",ecode);

						// Since loop is behind us just destroy the GPU context.
						cuCtxDestroy(context);
					}
					else errcuda("cuCtxCreate!",ecode);
				}
				else errcuda("cuDeviceGet!",ecode);
			}
			else err("selectedgpu is more of equal to number GPUs!");
		}
		else errcuda("cuDeviceGetCount!",ecode);
	}
	else errcuda("cuInit!",ecode);

	return 0;
}
