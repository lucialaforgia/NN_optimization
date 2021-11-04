//	Lucia La Forgia - 0000945383

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "hpc.h"

#define R 3
#define BLKDIM 128


float* generate_rand_array(int N) {
	/* returns an array of N random floats */
	float* rand_array = (float*)malloc(N*sizeof(float));
	for(int i=0; i<N; i++) {
		rand_array[i] = ((float)rand()/(float)(RAND_MAX))*3.0; // "*3.0" is to get more various floats
	}
	return rand_array;
}


float* init_inputs(int N) {
	/* returns input as an array of N random floats */
	return generate_rand_array(N);
}


float** init_weights(int N, int K) {
	/* returns weights as an array of K-1 arrays of random floats (one per output layer) */
	float** weights = (float**)malloc((K-1)*sizeof(float*));
	int output_neurons = N-(R-1); // neurons of the following output layer
	for(int i=0; i<K-1; i++){
		weights[i] = generate_rand_array(R*output_neurons); // weights of the layer i
		output_neurons -= R-1; // neurons of the following output layer
	}
	return weights;
}


float* init_biases(int N) {
	/* returns biases as an array of N random floats */
	return generate_rand_array(N);
}


__device__ float sigmoid(float f) {
	/* returns computation of the sigmoid function over a given float */
	return 1/(1+exp(-f));
}


//MY VERSION: COMPUTATION OF A SINGLE OUTPUT NEURON PER CUDA CORE
__global__ void compute_output(float* input, int N, float* weights, float bias, float* output) {
	/* computes the output layer using the input layer, the weights and the bias */
	__shared__ float shared_input[BLKDIM+R-1];
	__shared__ float shared_bias;
	const unsigned int global_i = blockIdx.x*blockDim.x+threadIdx.x;
	const unsigned int local_i = threadIdx.x;
	float temp;
	
	if (global_i<N) {
		shared_input[local_i] = input[global_i];  // initialise
		
		if (local_i<R-1 && global_i+BLKDIM<N) {
			shared_input[local_i+BLKDIM] = input[global_i+BLKDIM];  // initialise
		}
		if (local_i==0) {
			shared_bias = bias;  // initialise
		}
		__syncthreads(); // to avoid race conditions
		
		if(global_i<N-(R-1)) {
			temp = 0.0; // initialise
			for(int j=0; j<R; j++) { // iterates over the needed inputs and weights
				temp += shared_input[local_i+j]*weights[global_i*R+j];
			}
			output[global_i] = sigmoid(temp+shared_bias); // adding bias, applying the sigmoid function and saving
		}
	}
}


int main(int argc, char* argv[]) {
	// variables' declaration
	int N, K;
	float* h_bias;
	float* h_input;
	float** h_weights;
	float* h_output;
	float* d_input;
	float* d_weights;
	float* d_output;
	float* temp; //utility variable
	double time_start, time_stop;
	
	// getting size N of layer0 and number K of layers through command line
	if (argc != 3) {
		printf("INSERT TWO ARGUMENTS!");
		return EXIT_FAILURE;
	}
	N = atoi(argv[1]);
	K = atoi(argv[2]);
	if (N<=0 || K<=2 || N<=(K+1)*(R-1)) {
		printf("ILLEGAL ARGUMENTS!");
		return EXIT_FAILURE;
	}
	// initialise all
	srand(N);
	h_bias = init_biases(K-1);
	h_input = init_inputs(N);
	h_weights = init_weights(N, K);
	h_output = (float *)malloc((N-(R-1))*sizeof(float));
	
	// instantiating memory on device for input, weights and output
	cudaSafeCall(cudaMalloc((void **)&d_input, N*sizeof(float)));
	cudaSafeCall(cudaMalloc((void **)&d_weights, (N-(R-1))*R*sizeof(float)));
	cudaSafeCall(cudaMalloc((void **)&d_output, (N-(R-1))*sizeof(float)));
	
	
	time_start = hpc_gettime(); // saving starting time
	
	cudaSafeCall(cudaMemcpy(d_input, h_input, N*sizeof(float), cudaMemcpyHostToDevice)); // copying input to the device
	
	for(int i=0; i<K-1; i++) { // iterating along number of layers
		// copy weights of the current layer
		cudaSafeCall(cudaMemcpy(d_weights, h_weights[i], N*sizeof(float), cudaMemcpyHostToDevice));
		// compute output of the current layer
		compute_output<<<(N+BLKDIM-1)/BLKDIM, BLKDIM>>>(d_input, N, d_weights, h_bias[i], d_output);
		cudaCheckError(); // synchronization of cuda device
		if(i<K-2) {
			// using i layer's output as i+1 layer's input
			temp = d_input;
			d_input = d_output;
			d_output = temp;
			N -= R-1;
		}
	}

	cudaSafeCall(cudaMemcpy(h_output, d_output, (N-(R-1))*sizeof(float), cudaMemcpyDeviceToHost)); // copying final output to the host

	time_stop = hpc_gettime(); // saving stopping time

	
	// getting execution time
	printf("EXECUTION TIME: %f\n", time_stop-time_start);
	
	// deallocating memory
	cudaSafeCall(cudaFree(d_input));
	cudaSafeCall(cudaFree(d_weights));
	cudaSafeCall(cudaFree(d_output));
	free(h_bias);
	free(h_input);
	free(h_output);
	for(int i=0; i<K-1; i++) {
		free(h_weights[i]);
	}
	free(h_weights);
	
	
	return EXIT_SUCCESS;
}
