/**********************************************************
Name:	algorithm_CUDA_impl
Date:	20200529
Author: DC.Cheng
Note:	psuedo code release
Brief:	1. MovingAverageFiltering in CUDA in 2D / 3D
	2. psuedo code in gridSize in 3D 2D blockSize in 2D
Reference: 
	1. https://stackoverflow.com/questions/22577857/3d-convolution-with-cuda-using-shared-memory
	2. https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf
**********************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
__global__ void kernel(float *id, float *od, int w, int h, int depth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	const int dataTotalSize   = w * h * depth;				
	const int radius		  = 2;
	const int filter_size	  = 2*radius + 1;
	const int sW			  = 6;				/* sW == 2 * filter_radius + blockDim.x (or same as 2 * filter_radius + blockDim.y) */
	/* boarder do not concerned */
	if(x >= w || y >= h || z >= depth)	
		return;
	else
	{	
		//global defined 
		int idx = z*w*h+y*w+x;

		//3d grid(blocks) 2d block(threads)
		int threadsPerBlock = blockDim.x * blockDim.y;
		int blockId		    = blockIdx.x + blockIdx.y * gridDim.x
							+ gridDim.x * gridDim.y * blockIdx.z;									
		int threadId	    = (blockId * threadsPerBlock) 
							+ (threadIdx.y * blockDim.x) + threadIdx.x;	
		int g_Idx			= threadId;		

		//2d shared memory working
		__shared__ unsigned char smem[sW][sW];
		int s_Idx = threadIdx.x + (threadIdx.y * sW);
		int s_IdxY = s_Idx / sW;
		int s_IdxX = s_Idx % sW;	

		//Here: definition error, need edit, haven't finished yet.
		//int g_IdxY = s_IdxY + (blockIdx.y * blockDim.y);
		//int g_IdxX = s_IdxX + (blockIdx.x * blockDim.x);
		//int g_Idx  = g_IdxX + (g_IdxY * w);	
		
		//32 threads working together per warp
		if(s_IdxY < sW && s_IdxX < sW)	//Here: boarder concerned error, need edit
		{
			if(x >= 0 && y < w && y >= 0 && y < h && z >= 0 && z < depth )	//Here: boarder concerned error, need edit
				smem[s_IdxY][s_IdxX] = id[g_Idx];
			else
				smem[s_IdxY][s_IdxX] = 0;
			__syncthreads();
		}

		/*compute the sum using shared memory*/
		float avg = 0.0;
		for (int i = -radius; i <= radius; i++){
			if(s_IdxY + i < 0 /*|| g_IdxY > h*/ )			//Here: boarder concerned error, need edit
				avg += 0.0;
			else
				avg += smem[s_IdxY+i][s_IdxX];
		}

		/*register to global, by now thread*/ 
		avg /= filter_size;     
		if(idx < dataTotalSize)
			od[idx] = avg;	
	}
}

int main()
{
	//in
	float in[] =
    {
		1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f, 7.1f, 8.1f,
		1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f, 7.1f, 8.1f,
		1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f, 7.1f, 8.1f,
		1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f, 7.1f, 8.1f,
		1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f, 7.1f, 8.1f,
		1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f, 7.1f, 8.1f,
		1.2f, 2.2f, 3.2f, 4.2f, 5.2f, 6.2f, 7.2f, 8.2f,
		1.2f, 2.2f, 3.2f, 4.2f, 5.2f, 6.2f, 7.2f, 8.2f,
		1.2f, 2.2f, 3.2f, 4.2f, 5.2f, 6.2f, 7.2f, 8.2f,
		1.2f, 2.2f, 3.2f, 4.2f, 5.2f, 6.2f, 7.2f, 8.2f,
		1.2f, 2.2f, 3.2f, 4.2f, 5.2f, 6.2f, 7.2f, 8.2f,
		1.2f, 2.2f, 3.2f, 4.2f, 5.2f, 6.2f, 7.2f, 8.2f,
		1.3f, 2.3f, 3.3f, 4.3f, 5.3f, 6.3f, 7.3f, 8.3f,
		1.3f, 2.3f, 3.3f, 4.3f, 5.3f, 6.3f, 7.3f, 8.3f,
		1.3f, 2.3f, 3.3f, 4.3f, 5.3f, 6.3f, 7.3f, 8.3f,
		1.3f, 2.3f, 3.3f, 4.3f, 5.3f, 6.3f, 7.3f, 8.3f,
		1.3f, 2.3f, 3.3f, 4.3f, 5.3f, 6.3f, 7.3f, 8.3f,
		1.3f, 2.3f, 3.3f, 4.3f, 5.3f, 6.3f, 7.3f, 8.3f,
	};
	float* out;
	float* in_cu, *out_cu;
	int width  = 8;
	int height = 6;
	int depth  = 3;
	out = new float[width * height * depth];

	//processing
	cudaMalloc((void **)&in_cu, sizeof(float)* width * height * depth);
	cudaMalloc((void **)&out_cu, sizeof(float)* width * height * depth);
	cudaMemcpy(in_cu, in, sizeof(float)* width * height * depth, cudaMemcpyHostToDevice);
	const int kernel_filter_radius = 2;
	const int threadsNum = 2;
	const int threadsNumZ = 1;
	dim3 gridSize((width + threadsNum - 1) / threadsNum, (height + threadsNum - 1)/threadsNum, (depth + threadsNumZ - 1) / threadsNumZ); //(4,3,3)
	dim3 blockSize(threadsNum, threadsNum, threadsNumZ); //(2,2,1)
	kernel <<< gridSize, blockSize>>> (in_cu, out_cu, width, height, depth);	
	cudaMemcpy(out, out_cu, sizeof(float)* width * height * depth, cudaMemcpyDeviceToHost);
	
	//in
	for(int k=0; k<depth; k++){
		for(int j=0; j<height; j++){
			for(int i=0; i<width; i++){
				std::cout << in[k*width*height+j*width+i] << ",\t";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}
	std::cout << "==========seperate line ==========\n";

	//out
	for(int k=0; k<depth; k++){
		for(int j=0; j<height; j++){
			for(int i=0; i<width; i++){
				float dif = in[k*width*height+j*width+i] - out[k*width*height+j*width+i];
				if(dif != 0.0)
					std::cout << "error!" << ",\t";
				else
					std::cout << out[k*width*height+j*width+i] << ",\t";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}
	std::cout << "==========seperate line ==========\n";

	//delete
	delete[] out;
	cudaFree(in_cu);
	cudaFree(out_cu);
}
