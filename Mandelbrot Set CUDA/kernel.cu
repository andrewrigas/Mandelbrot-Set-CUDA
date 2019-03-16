#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "win-gettimeofday.h"
#include <stdio.h>

using namespace std;

/* Number of threads per block */
//Divide threads in block in 32 thdreads wrap for no divergent
#define TILE_WIDTH 32

//Resolution
#define WIDTH 1080 /* Width of mandelbrot set Matrix Image */
#define HEIGHT 1080 /* Height of mandelbrot set Matrix Image */

#define MaxRGB 256 //Max RGB value

//Data type declaration of a RGB
typedef struct {
	unsigned int red;
	unsigned int green;
	unsigned int blue;
} RGB;

//Data type declaration of a mandelbrot
typedef struct {
	RGB* image;
	unsigned int width;
	unsigned int height;
} Mandelbrot;

//Kenerl
__global__ void mandelbrotKernel(Mandelbrot mandelbrot, double* cr, double* ci) {
	//Minimize the global memory access by creating shared variables and store Ci and Cr values
	__shared__ double ci_s[HEIGHT];
	__shared__ double cr_s[WIDTH];
	// Row index to access to the rows of the product matrix (row-major convention)
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	// Col index to access to the rows of the product matrix (row-major convention)
	int col = blockIdx.x * blockDim.x + threadIdx.x;


	// If the dimensions of the matrices are not both multiples of the block size then some threads will not be computing elements of the product matrix
	if (row > mandelbrot.height || col > mandelbrot.width) return;

	int index = row * mandelbrot.width + col;

	//Store from global memory the C values to the shared memeory arrays
	ci_s[row] = ci[row];
	cr_s[col] = cr[col];

	/*mandelbrot Set Function and Calculations
	F(z) = z^2 + c
	Where c is a complex number
	Z start from 0 So f(z)1 = c;
	c = a + bi
	Where a and b are real numbers and i is an imaginari number
	(a + bi)^2 = a^2 + 2*abi + (bi)^2
	i is sqrt(-1) so sqrt(-1) ^ 2 = -1
	So z^2 = (a^2 + 2*bi - b)
	F(z)2 = (a^2 + 2*bi - b) + (a + bi)
	*/
	int i = 0;
	//Create variables to store the z Real values and Z imaginari values
	double zr = 0.0;
	double zi = 0.0;

	//How many times loop to find if number is increasing to infinite
	const int maxIterations = 500;

	//Check if number is incresing to infinite or end by the iterrations
	while (i < maxIterations && zr * zr + zi * zi < 4.0) {
		//Calculate fz by using the c real value
		double fz = zr * zr - zi * zi + cr_s[col];
		//Calculate the z imaginari value using the c imaginari value
		zi = 2.0 * zr * zi + ci_s[row];
		//Store new z real value to zr
		zr = fz;
		//+1 Next itteration
		i++;
	}

	//Create variables to store r g b values
	int r, g, b;

	//Colouring the mandelbrot set image
	int maxRGB = 256;
	int max3 = maxRGB * maxRGB * maxRGB;
	double t = (double)i / (double)maxIterations;
	i = (int)(t * (double)max3);
	b = i / (maxRGB * maxRGB);
	int nn = i - b * maxRGB * maxRGB;
	r = nn / maxRGB;
	g = nn - r * maxRGB;

	//Save Red Green Blue colours to our array of RGBS
	mandelbrot.image[index].red = r;
	mandelbrot.image[index].green = g;
	mandelbrot.image[index].blue = b;
}

// Initialize functions
int getCValues(double* c, int state, double beginRange, double endRange, double minVal, double maxVal);
cudaError_t mandelbrotSetWithCUDA(Mandelbrot mandelbrot, double* cr, double* ci);

int main(int argc, char* argv[])
{
	//Start timer Main the hole execution
	double startTimeMain = get_current_time();

	//Set default values to 0
	unsigned int width = 0;
	unsigned int height = 0;
	unsigned int maxRGB = 0;

	if (argc < 2) {
		width = WIDTH; //Set Width value
		height = HEIGHT; //Set Height value
		maxRGB = MaxRGB; //Set MaxRGB value
	}
	else {
		width = atoi(argv[1]); //Set Width value
		height = atoi(argv[2]); //Set Height value
		maxRGB = atoi(argv[3]); //Set MaxRGB Value
	}

	//Create an instance of a mandlerbort set image an array of rgb values
	Mandelbrot mandelbrot;

	//Create arrays that we will store the range values of real numbers and imaginari
	double* cr;
	double* ci;

	//Set the range of the mandelbrot set for c Real number and Imaginari values (Zoom in, Zoom out)
	double minValR = -2.0;
	double maxValR = 1.0;
	double minValI = -1.5;
	double maxValI = 1.5;

	size_t size;

	//Set width and height to out mandelbrot set
	mandelbrot.width = width;
	mandelbrot.height = height;

	//Dynamic allocate memory space for the size of the image on host
	size = width * height * sizeof(RGB);
	mandelbrot.image = (RGB*)malloc(size);

	//Dynamic allocate memory space for the size of the C values real numbers on host
	size = width * sizeof(double);
	cr = (double*)malloc(size);
	//Dynamic allocate memory space for the size of the C values imaginari numbers on host
	size = height * sizeof(double);
	ci = (double*)malloc(size);

	//Fill the c Values
	getCValues(cr, 0, 0, width, minValR, maxValR);
	getCValues(ci, 0, 0, height, minValI, maxValI);

	//Funciton that runs Cuda kernel and initialize vectors for GPU
	cudaError_t cudaStatus = mandelbrotSetWithCUDA(mandelbrot, cr, ci);

	//Check if kernel launched correctly
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel failed!");
	}
	else {
		//Start timer PPM FILE
		double startTimePPM = get_current_time();
		printf("Creating PPM image File...\n");

		//Create a PPM image file
		ofstream fout("output_image.ppm");
		//Set it to be a PPM file
		fout << "P3" << endl;
		//Set the Dimensions
		fout << mandelbrot.width << " " << mandelbrot.height << endl;
		//Max RGB Value
		fout << maxRGB << endl;

		//Fill the image with rgb values
		for (int h = 0; h < height; h++) {
			//Unrolling Loop check less and store tow elements per loop Width must be divisible by 2
			for (int w = 0; w < width; w += 2) {
				//Calculate Index
				int index = h * width + w;
				//Store in image every RGB pixel
				fout << mandelbrot.image[index].red << " " << mandelbrot.image[index].green << " " << mandelbrot.image[index].blue << " ";
				fout << mandelbrot.image[index + 1].red << " " << mandelbrot.image[index + 1].green << " " << mandelbrot.image[index + 1].blue << " ";
			}
			fout << endl;
		}
		fout.close();
		//End timer PPM file Created
		double endTimePPM = get_current_time();
		printf("CPU Time creating PPM file: %lfs\n", endTimePPM - startTimePPM);
		printf("Done!! The PPM image file is ready!\n");
	}

	//End timer Main the hole execution
	double endTimeMain = get_current_time();

	printf("Time for the hole execution: %lfs\n", endTimeMain- startTimeMain);

	return 0;
}

//Function the fill the c values Recursively
int getCValues(double* c, int state, double beginRange, double endRange, double minVal, double maxVal) {
	//Check if we are on the last state on end Range
	if (state < endRange) {
		//Fill c values by breaking into equal part the range between minVal and maxVal by the state and the range between Begin and End
		c[state] = ((state - beginRange) / (endRange - beginRange))*(maxVal - minVal) + minVal;
		//call the functio itself
		return getCValues(c, state + 1, beginRange, endRange, minVal, maxVal);
	}
	else {
		//return 0 for success
		return 0;
	}
}

cudaError_t mandelbrotSetWithCUDA(Mandelbrot mandelbrot, double* cr, double* ci)
{
	//Check status
	cudaError_t cudaStatus;

	//Create double variables to store the start time and end time with data transfer
	double startTimeData;
	double endTimeData;

	//Create double variables to store the start time and end time without data transfer
	double startTimeNoData;
	double endTimeNoData;

	//Store width and height
	unsigned int width = mandelbrot.width;
	unsigned int height = mandelbrot.height;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	//Create an instance of mandelbrot
	Mandelbrot mandelbrot_d;
	//Set width and height to the mandelboer instance
	mandelbrot_d.width = width;
	mandelbrot_d.height = height;
	//Abount of RGB memory needed to alocate memory of the device
	size_t  mandlebortSize = width * height * sizeof(RGB);
	//Dynamic allocate memory space for the mandelbrot instance on device
	cudaStatus = cudaMalloc((void **)&mandelbrot_d.image, mandlebortSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed mandelbrot_d on device!");
	}

	//Create cr vector on device
	double* cr_d;
	//Abount of double bytes memory needed to alocate memory of the device
	size_t CRealSize = width * sizeof(double);
	//Dynamic allocate memory space for store c real number values on device
	cudaStatus = cudaMalloc((void**)&cr_d, CRealSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed C Real on device!");
	}

	//Create cr and ci to store the c value of our mandelrbort set on device
	double* ci_d;
	// Abount of double bytes memory needed to alocate memory of the device
	size_t  CImagSize = height * sizeof(double);
	//Dynamic allocate memory space for store c imaginari number values on device
	cudaStatus = cudaMalloc((void**)&ci_d, CImagSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed C Imaginari on device!");
	}

	//Start timer with data transfer
	startTimeData = get_current_time();

	//Copy input C real vector memory from host to device
	cudaStatus = cudaMemcpy(cr_d, cr, CRealSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed C real host to device!");
	}

	//Copy input C imaginari vector memory from host to device
	cudaStatus = cudaMemcpy(ci_d, ci, CImagSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed C imaginari host to device!");
	}


	//If tile_width doesnt divide width or height exactly
	//We use the formula + tile_width - -1 so if it was divided exactly will stay the same
	//If not we will use some extra threads to do the calculations and some of them will do nothing
	int blocks_x = (width + TILE_WIDTH - 1) / TILE_WIDTH;
	int blocks_y = (height + TILE_WIDTH - 1) / TILE_WIDTH;

	//Create 2D grid and 2D block dimensions
	dim3 dimGrid(blocks_x, blocks_y, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

	printf("Launching Kernel...\n");

	//Start timer without data transfer
	startTimeNoData = get_current_time();

	//Launch Kernel
	mandelbrotKernel << <dimGrid, dimBlock >> > (mandelbrot_d, cr_d, ci_d);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "mandelbrotKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching mandelbrotKernel!\n", cudaStatus);
		goto DeallocateMemory;
	}

	//End timer without data transfer
	endTimeNoData = get_current_time();

	printf("Kernel Finished\n");

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(mandelbrot.image, mandelbrot_d.image, mandlebortSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	//End timer with data transfer
	endTimeData = get_current_time();


	printf("GPU Time (NOT Including Data Transfer): %lfs\n", endTimeNoData - startTimeNoData);
	printf("GPU Time (Including Data Transfer): %lfs\n", endTimeData - startTimeData);

	//Free allocated memory
DeallocateMemory:
	cudaFree(mandelbrot_d.image);
	cudaFree(cr_d);
	cudaFree(ci_d);
	return cudaStatus;
}
