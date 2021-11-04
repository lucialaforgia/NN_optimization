//	Lucia La Forgia - 0000945383

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define R 3


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


float sigmoid(float f) {
	/* returns computation of the sigmoid function over a given float */
	return 1/(1+exp(-f));
}


//MY VERSION: PARALLELIZATION OF THE FOR-LOOP ITERATING THE OUTPUT NEURONS
void compute_output(float* input, int N, float* weights, float bias, float* output) {
	/* computes the output layer using the input layer, the weights and the bias */
	int i;
	int output_neurons = N-(R-1);
	#pragma omp parallel for default(shared) schedule(static) num_threads(omp_get_max_threads())
	for(i=0; i<output_neurons; i++){ // iterates over output
		output[i] = 0.0; // initialise
		for(int j=0; j<R; j++){ // iterates over the needed inputs and weights
			output[i] += input[i+j]*weights[i*R+j];
		}
		output[i] = sigmoid(output[i]+bias); // adding bias and applying the sigmoid function
	}
}


int main(int argc, char* argv[]) {
	// variables' declaration
	int N, K;
	float* bias;
	float* input;
	float** weights;
	float* output;
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
	bias = init_biases(K-1);
	input = init_inputs(N);
	weights = init_weights(N, K);
	output = (float *)malloc(N*sizeof(float));
	
	
	time_start = omp_get_wtime(); // saving starting time

	for(int i=0; i<K-1; i++) { // iterating along number of layers
		compute_output(input, N, weights[i], bias[i], output);
		if(i<K-2) {
			// using i layer's output as i+1 layer's input
			temp = input;
			input = output;
			output = temp;
			N -= R-1;
		}
	}

	time_stop = omp_get_wtime(); // saving stopping time

	
	// getting execution time
	printf("EXECUTION TIME: %f\n", time_stop-time_start);
	
	// deallocating memory
	free(bias);
	free(input);
	free(output);
	for(int i=0; i<K-1; i++) {
		free(weights[i]);
	}
	free(weights);
	
	
	return EXIT_SUCCESS;
}
