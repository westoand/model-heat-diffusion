#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "math.h"

using namespace std;

#define BLOCK_SIZE 32

/* Helper kernel to set initial room state */
__global__ void
cu_initialize_room(float *room_state_d, const int ARRAY_SIZE,
                   const int HEATER_START, const int HEATER_END);

/* Helper kernel to compute room state after one additional time step */
__global__ void
cu_room_diffusion_step(float *room_state_d_curr,
                       float *room_state_d_prev, const int TILES,
                       const int X_SIZE, const int Y_SIZE,
                       const int HEATER_START, const int HEATER_END);

/* Helper function to compute the room's temperature */
void calc_room_diffusion(float *temperature, const int TILES,
                         const int TIME_STEPS,
                         const int LOC_X,
                         const int X_SIZE,
                         const int LOC_Y,
                         const int Y_SIZE,
                         float *room_state1,
                         float *room_state2);


/* Helper function to print the room's state */
void print_room_state(const float *state, const int SIZE,
                      const int X_SIZE, const int Y_SIZE);

/* Model the heater at consistent 100 degrees */
const int HEATER_TEMP = 100;

/* Model the room at 23 degrees at initialization */
const int INIT_TEMP = 23;


/**********************************************************************
 * This function serves as the program "driver" for an implementation
 * of a heat diffusion model of a room.
 *
 * @param argc is the number of command-line arguments.
 * @param v is the array of strings representing the command-line args.
 *********************************************************************/
int main(int argc, char *argv[]) {
    if (argc != 8) {
        cerr <<
             "usage: programName slice timeSteps locX locY xSize ySize"
             << endl;
        exit(-1);
    }
    auto start_cuda = chrono::high_resolution_clock::now();

    const int TILES = stoi(argv[2]);
    const int TIME_STEPS = stoi(argv[3]);
    const int LOC_X_INDEX = stoi(argv[4]);
    const int LOC_Y_INDEX = stoi(argv[5]);
    const int X_SIZE = stoi(argv[6]);
    const int Y_SIZE = stoi(argv[7]);
    if (X_SIZE * Y_SIZE != TILES)
        cerr << "Invalid X or Y Size dimensions (X x Y must = TILES)"
             << endl;
    float temperature = -1.0;

    //make space for copying back room state
//    auto *room_state1 = (float *) malloc(sizeof(float) * TILES);
//    auto *room_state2 = (float *) malloc(sizeof(float) * TILES);
    auto *room_state1 = (float *) malloc(sizeof(float) * 1);
    auto *room_state2 = (float *) malloc(sizeof(float) * 1);
    calc_room_diffusion(&temperature, TILES, TIME_STEPS, LOC_X_INDEX,
                        X_SIZE, LOC_Y_INDEX, Y_SIZE, room_state1,
                        room_state2);
    //print the proper half of the room state
//    if ((TIME_STEPS + 1) % 2 == 0)
//        print_room_state(room_state1, TILES, X_SIZE, Y_SIZE);
//    else
//        print_room_state(room_state2, TILES, X_SIZE, Y_SIZE);

    auto end_cuda = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> seq_millisec = end_cuda -
                                                   start_cuda;
    cout << "Temp: " << temperature << endl;
    cout << "Time: " << seq_millisec.count() << " ms" << endl;
}


/**********************************************************************
 * This helper function computes the room's temperature at the
 * specified location after the specified number of time steps.
 *
 * @param temperature is the location to update with the results of
 * the computation.
 * @param TILES is the number of pieces the room is divided into
 * for temperature computations.
 * @param TIME_STEPS is the total number of time steps to compute.
 * @param LOC_X is the location on the room to be sampled.
 * @param X_SIZE is the room width dimension.
 * @param LOC_Y is the location on the room to be sampled.
 * @param Y_SIZE is the room depth dimension.
 * @param room_state1 is the first copy of the room state.
 * @param room_state2 is the second copy of the room state.
 *********************************************************************/
void calc_room_diffusion(float *temperature, const int TILES,
                         const int TIME_STEPS,
                         const int LOC_X,
                         const int X_SIZE,
                         const int LOC_Y,
                         const int Y_SIZE,
                         float *room_state1,
                         float *room_state2) {
    float *room_state1_d;
    float *room_state2_d;
    cudaError_t result;

    result = cudaMalloc((void **) &room_state1_d,
                        sizeof(float) * TILES);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaMalloc (block) failed.");
        exit(1);
    }

    result = cudaMalloc((void **) &room_state2_d,
                        sizeof(float) * TILES);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaMalloc (block) failed.");
        exit(1);
    }

    dim3 dimblock(BLOCK_SIZE, BLOCK_SIZE);
    int nBlocks = ceil(((double) (TILES)) / BLOCK_SIZE);
    dim3 dimgrid(nBlocks, nBlocks);

    //initialization - assume heater occupies middle third of  wall
    const int HEATER_START = X_SIZE / 3;
    const int HEATER_END = HEATER_START * 2;
    cu_initialize_room <<<dimgrid, dimblock>>>
            (room_state1_d, TILES, HEATER_START, HEATER_END);
    cu_initialize_room <<<dimgrid, dimblock>>>
            (room_state2_d, TILES, HEATER_START, HEATER_END);

    for (int k = 0; k < TIME_STEPS; k++) {
        if (k % 2 == 0)
            cu_room_diffusion_step<<<dimgrid, dimblock>>>
                    (room_state1_d, room_state2_d, TILES,
                     X_SIZE, Y_SIZE,
                     HEATER_START, HEATER_END);
        else
            cu_room_diffusion_step<<<dimgrid, dimblock>>>
                    (room_state2_d, room_state1_d, TILES,
                     X_SIZE, Y_SIZE,
                     HEATER_START, HEATER_END);
    }

    // transfer whole room state 1 back to host during dev
    //result = cudaMemcpy(room_state1, room_state1_d,
    // sizeof(float) * TILES, cudaMemcpyDeviceToHost);

    // transfer only resulting temp back to host
    if (TIME_STEPS % 2 == 1)
        result = cudaMemcpy(temperature,
                            (void *) &room_state1_d
                            [(LOC_Y * X_SIZE) + LOC_X],
                            sizeof(float),
                            cudaMemcpyDeviceToHost);
    else
        result = cudaMemcpy(temperature,
                            (void *) &room_state2_d
                            [(LOC_Y * X_SIZE) + LOC_X],
                            sizeof(float),
                            cudaMemcpyDeviceToHost);

    if (result != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy host <- dev (block) failed.");
        exit(1);
    }
    // transfer whole room state 2 back to host during dev
//    result = cudaMemcpy(room_state2,
//    room_state2_d, sizeof(float) * TILES, cudaMemcpyDeviceToHost);
//    if (result != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy host <- dev (block) failed.");
//        exit(1);
//    }
    result = cudaFree(room_state1_d);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaFree (block) failed.");
        exit(1);
    }
    result = cudaFree(room_state2_d);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaFree (block) failed.");
        exit(1);
    }
}


/**********************************************************************
 * This kernel computes the room's temperature at the specified
 * location after one additional time step.
 *
 * @param room_state_d_curr is the "next" copy of the room state.
 * @param room_state_d_prev is the "previous" copy of the room state.
 * @param TILES is the number of pieces the room is divided into
 * for temperature computations.
 * @param X_SIZE is the room width dimension.
 * @param Y_SIZE is the room depth dimension.
 * @param HEATER_START is the X index of the first heater tile.
 * @param HEATER_END is the X index after the last heater tile.
 *********************************************************************/
__global__ void
cu_room_diffusion_step(float *room_state_d_curr,
                       float *room_state_d_prev, const int TILES,
                       const int X_SIZE,
                       const int Y_SIZE, const int HEATER_START,
                       const int HEATER_END) {
    int el = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int i = el / X_SIZE;
    int j = (blockIdx.x * BLOCK_SIZE + threadIdx.x) % X_SIZE;
    const int HEATER = 100.0;
    if (el < TILES) {
        //case 1 - heater tiles
        if ((i == 0) && ((j >= HEATER_START) && (j < HEATER_END))) {
            //ignore i since we know we're in the first row
            room_state_d_curr[el] = HEATER;
        }
            //case 2 - NW corner
        else if ((i == 0) && (j == 0)) {
            //average myself, E, and S from last time step
            room_state_d_curr[el] = (room_state_d_prev[el] +
                                     room_state_d_prev[(el) +
                                                       1] +
                                     room_state_d_prev[(el) +
                                                       X_SIZE]) / 3;
        }
            //case 3 - NE corner
        else if ((i == 0) && (j == X_SIZE - 1)) {
            //average myself, W and S from last time step
            room_state_d_curr[el] =
                    ((float) (room_state_d_prev[el] +
                              room_state_d_prev[el -
                                                1] +
                              room_state_d_prev[el +
                                                X_SIZE])) / 3;
        }
            //case 4 - N border
        else if (i == 0) {
            //average myself, W, E, and S from last time step
            room_state_d_curr[el] =
                    (room_state_d_prev[el] +
                     room_state_d_prev[el - 1] +
                     room_state_d_prev[el + 1] +
                     room_state_d_prev[el + X_SIZE]) /
                    4;
        }
            //case 5 - SW corner
        else if ((i == Y_SIZE - 1) && (j == 0)) {
            //average myself, E, and N from last time step
            room_state_d_curr[el] =
                    (room_state_d_prev[el] +
                     room_state_d_prev[el +
                                       1] +
                     room_state_d_prev[el -
                                       X_SIZE]) / 3;
        }
            //case 6 - W border
        else if (j == 0) {
            //average myself, E, N, and S from last time step
            room_state_d_curr[el] =
                    (room_state_d_prev[el] +
                     room_state_d_prev[el +
                                       1] +
                     room_state_d_prev[el -
                                       X_SIZE] +
                     room_state_d_prev[el +
                                       X_SIZE]) / 4;
        }
            //case 7 - SE corner
        else if ((i == Y_SIZE - 1) && (j == X_SIZE - 1)) {
            //average myself, W, and N from last time step
            room_state_d_curr[el] =
                    (room_state_d_prev[el] +
                     room_state_d_prev[el -
                                       1] +
                     room_state_d_prev[el -
                                       X_SIZE]) / 3;
        }
            //case 8 - E border
        else if (j == X_SIZE - 1) {
            //average myself, W, N, and S from last time step
            room_state_d_curr[el] =
                    (room_state_d_prev[el] +
                     room_state_d_prev[el -
                                       1] +
                     room_state_d_prev[el -
                                       X_SIZE] +
                     room_state_d_prev[el +
                                       X_SIZE]) / 4;
        }
            //case 8 - S border
        else if (i == Y_SIZE - 1) {
            //average myself, W, E, and N from last time step
            room_state_d_curr[el] =
                    (room_state_d_prev[el] +
                     room_state_d_prev[el - 1] +
                     room_state_d_prev[el + 1] +
                     room_state_d_prev[el - X_SIZE]) /
                    4;
        }
            //case 9 - common case of center of the room
        else {
            //average W, E, N, and S from last time step
            room_state_d_curr[el] =
                    (room_state_d_prev[el - 1] +
                     room_state_d_prev[el + 1] +
                     room_state_d_prev[el - X_SIZE] +
                     room_state_d_prev[el + X_SIZE]
                    ) / 4;
        }
    }
}


/**********************************************************************
 * This helper function prints half of the room state in a "human-
 * readable format" (only one time step is printed).
 *
 * @param state is the pointer to the rod model's temperature state.
 * @param SIZE is the number of pieces the room is divided into
 * for temperature computations.
 * @param X_SIZE is the room width dimension.
 * @param Y_SIZE is the room depth dimension.
 *********************************************************************/
void print_room_state(const float *state, const int SIZE,
                      const int X_SIZE, const int Y_SIZE) {
    for (int i = 0; i < Y_SIZE; i++) {
        for (int j = 0; j < X_SIZE; j++) {
            cout << state[(i * X_SIZE) + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}


/**********************************************************************
 * This kernel initializes the room's temperature.
 *
 * @param room_state_d is the copy of the room state.
 * @param TILES is the number of pieces the room is divided into
 * for temperature computations..
 * @param HEATER_START is the X index of the first heater tile.
 * @param HEATER_END is the X index after the last heater tile.
 *********************************************************************/
__global__ void
cu_initialize_room(float *room_state_d, const int TILES,
                   const int HEATER_START, const int HEATER_END) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < TILES) {
        if (i >= HEATER_START && i < HEATER_END) {
            room_state_d[i] = HEATER_TEMP;
        } else room_state_d[i] = INIT_TEMP;
    }
}