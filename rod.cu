#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "math.h"

using namespace std;

#define BLOCK_SIZE 32

/* Helper kernel to set initial rod state */
__global__ void cu_initialize_rod(float *rod_state_d,
                                  const int ARRAY_SIZE);

/* Helper kernel to compute rod state after one additional time step */
__global__ void cu_rod_diffusion_step(float *rod_state_d,
                                      const int ARRAY_SIZE,
                                      const int START,
                                      const int END);

/* Helper function to compute the rod's temperature */
void calc_rod_diffusion(float *temperature, const int SLICES,
                        const int TIME_STEPS,
                        const int LOC_X,
                        const int X_SIZE, float *rod_state,
                        const int ARRAY_SIZE);

/* Helper function to print the rod's state */
void print_rod_state(const float *state, const int SIZE,
                     const int TIME_STEPS);

/* Model the bucket/heater at consistent 100 degrees */
const int HEATER_TEMP = 100;

/* Model the rod at 23 degrees at initialization */
const int INIT_TEMP = 23;


/**********************************************************************
 * This function serves as the program "driver" for an implementation
 * of a heat diffusion model of a rod.
 *
 * @param argc is the number of command-line arguments.
 * @param v is the array of strings representing the command-line args.
 *********************************************************************/
int main(int argc, char *argv[]) {
    if (argc != 8) {
        cerr << "usage: pgName slice timeSteps locX locY xSize ySize"
             << endl;
        exit(-1);
    }

    auto start_cuda = chrono::high_resolution_clock::now();

    const int SLICES = stoi(argv[2]);
    const int TIME_STEPS = stoi(argv[3]);
    const int LOC_X_INDEX = stoi(argv[4]);
    const int LOC_Y_INDEX = stoi(argv[5]);
    const int X_SIZE = stoi(argv[6]);
    const int Y_SIZE = stoi(argv[7]);
    float temperature = -1.0;
    const int COPIES = 2;
    //COPIES 2 extra spaces to handle the bucket initialization
    // in both copies
    const int ARRAY_SZ = (SLICES * COPIES) + COPIES;

    //FIXME rm from performance trials
    //auto *rod_state = (float *) malloc(sizeof(float) * ARRAY_SZ);
    auto *rod_state = (float *) malloc(sizeof(float) * 1);
    calc_rod_diffusion(&temperature, SLICES, TIME_STEPS, LOC_X_INDEX,
                       X_SIZE, rod_state, ARRAY_SZ);
    //print_rod_state(rod_state, ARRAY_SZ, TIME_STEPS);

    auto end_cuda = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> seq_millisec = end_cuda -
                                                   start_cuda;

    cout << "Temp: " << temperature << endl;
    cout << "Time: " << seq_millisec.count() << " ms" << endl;
}


/**********************************************************************
 * This helper function computes the rod's temperature at the specified
 * location after the specified number of time steps.
 *
 * @param temperature is the location to update with the results of
 * the computation.
 * @param SLICES is the number of pieces the rod is divided into
 * for temperature computations.
 * @param TIME_STEPS is the total number of time steps to compute.
 * @param LOC_X is the location on the rod to be sampled.
 * @param rod_state is the location to copy the rod state.
 * @param ARRAY_SIZE is the total size of the rod array.
 *********************************************************************/
void calc_rod_diffusion(float *temperature, const int SLICES,
                        const int TIME_STEPS,
                        const int LOC_X,
                        const int X_SIZE, float *rod_state,
                        const int ARRAY_SIZE) {
    float *rod_state_d;
    cudaError_t result;
    const int CP_INDEX = (TIME_STEPS + 1) % 2;
    const int BASE = CP_INDEX * (ARRAY_SIZE / 2) + 1;


    result = cudaMalloc((void **) &rod_state_d,
                        sizeof(float) * ARRAY_SIZE);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaMalloc (block) failed.");
        exit(1);
    }

    dim3 dimblock(BLOCK_SIZE);
    dim3
    dimgrid(ceil((double) (ARRAY_SIZE) / BLOCK_SIZE));

    cu_initialize_rod <<<dimgrid, dimblock>>>(rod_state_d, ARRAY_SIZE);

    for (int k = 0; k < TIME_STEPS; k++) {
        const int START = ((k % 2) * ((ARRAY_SIZE / 2) + 1));
        const int END = ((((k) % 2) + 1) * (ARRAY_SIZE / 2)) - 1;
        cu_rod_diffusion_step<<<dimgrid, dimblock>>>(rod_state_d,
                                                     ARRAY_SIZE,
                                                     START,
                                                     END);
    }

    // transfer whole rod state back to host
    //result = cudaMemcpy(rod_state, rod_state_d,
    // sizeof(float) * ARRAY_SIZE, cudaMemcpyDeviceToHost);
    // transfer only resulting temp back to host
    result = cudaMemcpy(temperature,
                        (void *) &rod_state_d[BASE + LOC_X],
                        sizeof(float), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy host <- dev (block) failed.");
        exit(1);
    }

    result = cudaFree(rod_state_d);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaFree (block) failed.");
        exit(1);
    }
}


/**********************************************************************
 * This kernel computes the rod's temperature at the specified
 * location after one additional time step.
 *
 * @param rod_state_d is the location of the rod state.
 * @param ARRAY_SIZE is the total size of the rod array.
 * @param START is the location to begin computing the next time step.
 * @param START is the location to end computing the next time step.
 *********************************************************************/
__global__ void cu_rod_diffusion_step(float *rod_state_d,
                                      const int ARRAY_SIZE,
                                      const int START, const int END) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int PREV = ((ARRAY_SIZE / 2) + i) % ARRAY_SIZE;
    if (i >= START && i < END) {
        //start of both copies is the 100 degree const so do nothing
        if ((i == 0) || (i == (((ARRAY_SIZE / 2)))));
            //common case
        else {
            rod_state_d[i] = ((float)
                    (rod_state_d[PREV - 1] + rod_state_d[PREV + 1]))
                             / 2;
        }
    }
    //special case of end of the rod
    if (i == END)
        rod_state_d[i] = ((float)
                (rod_state_d[PREV - 1] + rod_state_d[PREV]))
                         / 2;
}


/**********************************************************************
 * This kernel initializes the rod's temperature.
 *
 * @param rod_state_d is the location of the rod state.
 * @param ARRAY_SIZE is the total size of the rod array.
 *********************************************************************/
__global__ void cu_initialize_rod(float *rod_state_d,
                                  const int ARRAY_SIZE) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < ARRAY_SIZE) {
        if ((i == 0) || (i == (ARRAY_SIZE / 2)))
            rod_state_d[i] = HEATER_TEMP;
        else {
            rod_state_d[i] = INIT_TEMP;
        }
    }
}


/**********************************************************************
 * This helper function prints half of the rod state in a "human-
 * readable format" (only one time step is printed).
 *
 * @param state is the pointer to the rod model's temperature state.
 * @param SIZE is the number of pieces the rod is divided into
 * for temperature computations.
 * @param TIME_STEPS is the time step to display.
 *********************************************************************/
void print_rod_state(const float *state, const int SIZE,
                     const int TIME_STEPS) {
    for (int i = 1; i < SIZE / 2; i++) {
        cout << state[(((TIME_STEPS + 1) % 2) * (SIZE / 2)) + i]
             << " ";
    }
    cout << endl;
}