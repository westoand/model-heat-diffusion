#include <iostream>
#include <chrono>

/* Module for computing diffusion in a rod */
float diffusion_seq_1D(char *const *argv);

/* Module for computing diffusion in a room */
float diffusion_seq_2D(char *const *argv);

/* Helper function for printing a time step of the rod model */
void printRodTimeStep(const float *rod, int SLICES, int X_SIZE,
                      int TIME_STEP);

/* Helper function for printing all of the room model's state */
void printArray(const float *room, int TILES, int X_SIZE, int Y_SIZE);

/* Helper function for printing a time step of the room model */
void printRoomTimeStep(const float *room, int TILES, int X_SIZE,
                       int Y_SIZE, int TIME_STEPS);


using namespace std;


/**********************************************************************
 * This function serves as the program "driver" for an implementation
 * of a heat diffusion model of a rod or a room.
 *
 * @param argc is the number of command-line arguments.
 * @param v is the array of strings representing the command-line args.
 *********************************************************************/
int main(int argc, char **argv) {
    auto start_seq = chrono::high_resolution_clock::now();
    const int DIMENSIONS = stoi(argv[1]);
    float temperature;
    switch (DIMENSIONS) {
        case 1:
            temperature = diffusion_seq_1D(argv);
            break;
        case 2:
            temperature = diffusion_seq_2D(argv);
            break;
        default:
            cerr << "Invalid dimension arg must be 1 or 2)" << endl;
    }
    auto end_seq = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> seq_millisec = end_seq - start_seq;
    cout << "Temp: " << temperature << endl;
    cout << "Time: " << seq_millisec.count() << " ms" << endl;
    return 0;
}


/**********************************************************************
 * This helper function serves as the program implementation
 * of a heat diffusion model for a rod.
 *
 * @param v is the array of strings representing the command-line args
 *
 * @return is the temperature value for the specified rod location.
 *********************************************************************/
float diffusion_seq_1D(char *const *argv) {
    const int SLICES = stoi(argv[2]);
    const int TIME_STEPS = stoi(argv[3]);
    const int LOC_X_INDEX = stoi(argv[4]);
    const int LOC_Y_INDEX = stoi(argv[5]);
    const int X_SIZE = stoi(argv[6]);
    const int Y_SIZE = stoi(argv[7]);

    //malloc 2 rods of space for "old" and "new" vals
    const int COPIES = 2;
    const int ARRAY_SZ = SLICES * COPIES;
    const int INIT_TEMP = 23.0;
    const int HEATER = 100.0;

    auto *rod = (float *) malloc(sizeof(float) * ARRAY_SZ);
    for (int i = 0; i < ARRAY_SZ; i++)
        rod[i] = INIT_TEMP;

    for (int i = 0; i < TIME_STEPS; i++) {
        const int CP_INDEX = ((i % 2) * SLICES);
        //calc each rod slice element
        for (int j = 0; j < SLICES; j++) {
            // the rod slice element in "this" time step
            const int EL_INDEX = CP_INDEX + j;
            // the rod slice element in "previous" time step
            const int PREV_INDEX = ((((i % 2) + 1) % 2) * SLICES) + j;
            //special case of first slice
            if (j == 0) {
                // avg of heater and right neighbor
                rod[EL_INDEX] = (HEATER + rod[PREV_INDEX + 1]) / 2;
            }
                //special case of last slice
                // -> average me and my left neighbor
            else if (j == SLICES - 1) {
                rod[EL_INDEX] =
                        (rod[PREV_INDEX - 1] + rod[PREV_INDEX]) / 2;
            }
                // common case - average left and right neighbors
                // from last time step
            else {
                rod[EL_INDEX] =
                        (rod[PREV_INDEX - 1] + rod[PREV_INDEX + 1])
                        / 2;
            }
        }
        //printRodTimeStep(rod, SLICES, X_SIZE, i);
    }
    const int CP_INDEX = (((TIME_STEPS + 1) % 2) * SLICES);
    return rod[CP_INDEX + LOC_X_INDEX];
}


/**********************************************************************
 * This helper function serves as the program implementation
 * of a heat diffusion model for a room.
 *
 * @param v is the array of strings representing the command-line args
 *
 * @return is the temperature value for the specified room location.
 *********************************************************************/
float diffusion_seq_2D(char *const *argv) {
    const int TILES = stoi(argv[2]);
    const int TIME_STEPS = stoi(argv[3]);
    const int LOC_X_INDEX = stoi(argv[4]);
    const int LOC_Y_INDEX = stoi(argv[5]);
    const int X_SIZE = stoi(argv[6]);
    const int Y_SIZE = stoi(argv[7]);
    if (X_SIZE * Y_SIZE != TILES)
        cerr << "Invalid X or Y Size dimensions (X x Y must = TILES)"
             << endl;
    //malloc 2 grids of space for "old" and "new" vals
    const int COPIES = 2;
    const int ARRAY_SZ = TILES * COPIES;
    const float INIT_TEMP = 23.0;
    const float HEATER = 100.0;

    auto *room = (float *) malloc(sizeof(float) * ARRAY_SZ);

    //initialization - assume heater occupies the middle third of wall
    const int HEATER_START = X_SIZE / 3;
    const int HEATER_END = HEATER_START * 2;
    for (int k = 0; k < COPIES; k++) {
        for (int i = 0; i < Y_SIZE; i++) {
            for (int j = 0; j < X_SIZE; j++) {
                //set heater tiles
                if ((i == 0) &&
                    ((j >= HEATER_START) &&
                     j < HEATER_END))
                    room[(k * TILES) + j] = HEATER;
                    //set common case tiles
                else room[(k * TILES) + (i * X_SIZE) + j] = INIT_TEMP;
            }
        }
    }

    //printArray(room, TILES, X_SIZE, Y_SIZE);
    for (int k = 0; k < TIME_STEPS; k++) {
        for (int i = 0; i < Y_SIZE; i++) {
            for (int j = 0; j < X_SIZE; j++) {
                const int EL_INDEX = ((k % 2) * (TILES)) +
                                     (i * X_SIZE) + j;
                const int PREV_EL_INDEX =
                        ((((k % 2) + 1) % 2) * (TILES)) +
                        (i * X_SIZE) + j;
                //case 1 - heater tiles
                if ((i == 0) && ((j >= HEATER_START) &&
                                 (j < HEATER_END))) {
                    //ignore i since we know we're in the first row
                    room[EL_INDEX] = HEATER;
                }
                    //case 2 - NW corner
                else if ((i == 0) && (j == 0)) {
                    //average myself, E, and S from last time step
                    room[EL_INDEX] = (room[PREV_EL_INDEX] +
                                      room[(PREV_EL_INDEX) +
                                           1] +
                                      room[(PREV_EL_INDEX) +
                                           X_SIZE]) / 3;
                }
                    //case 3 - NE corner
                else if ((i == 0) && (j == X_SIZE - 1)) {
                    //average myself, W and S from last time step
                    room[EL_INDEX] =
                            ((float) (room[PREV_EL_INDEX] +
                                      room[PREV_EL_INDEX -
                                           1] +
                                      room[PREV_EL_INDEX +
                                           X_SIZE])) / 3;
                }
                    //case 4 - N border
                else if (i == 0) {
                    //average myself, W, E, and S from last time step
                    room[EL_INDEX] =
                            (room[PREV_EL_INDEX] +
                             room[PREV_EL_INDEX - 1] +
                             room[PREV_EL_INDEX + 1] +
                             room[PREV_EL_INDEX + X_SIZE]) /
                            4;
                }
                    //case 5 - SW corner
                else if ((i == Y_SIZE - 1) && (j == 0)) {
                    //average myself, E, and N from last time step
                    room[EL_INDEX] =
                            (room[PREV_EL_INDEX] +
                             room[PREV_EL_INDEX +
                                  1] +
                             room[PREV_EL_INDEX -
                                  X_SIZE]) / 3;
                }
                    //case 6 - W border
                else if (j == 0) {
                    //average myself, E, N, and S from last time step
                    room[EL_INDEX] =
                            (room[PREV_EL_INDEX] +
                             room[PREV_EL_INDEX +
                                  1] +
                             room[PREV_EL_INDEX -
                                  X_SIZE] +
                             room[PREV_EL_INDEX +
                                  X_SIZE]) / 4;
                }
                    //case 7 - SE corner
                else if ((i == Y_SIZE - 1) && (j == X_SIZE - 1)) {
                    //average myself, W, and N from last time step
                    room[EL_INDEX] =
                            (room[PREV_EL_INDEX] +
                             room[PREV_EL_INDEX -
                                  1] +
                             room[PREV_EL_INDEX -
                                  X_SIZE]) / 3;
                }
                    //case 8 - E border
                else if (j == X_SIZE - 1) {
                    //average myself, W, N, and S from last time step
                    room[EL_INDEX] =
                            (room[PREV_EL_INDEX] +
                             room[PREV_EL_INDEX -
                                  1] +
                             room[PREV_EL_INDEX -
                                  X_SIZE] +
                             room[PREV_EL_INDEX +
                                  X_SIZE]) / 4;
                }
                    //case 8 - S border
                else if (i == Y_SIZE - 1) {
                    //average myself, W, E, and N from last time step
                    room[EL_INDEX] =
                            (room[PREV_EL_INDEX] +
                             room[PREV_EL_INDEX - 1] +
                             room[PREV_EL_INDEX + 1] +
                             room[PREV_EL_INDEX - X_SIZE]) /
                            4;
                }
                    //case 9 - common case of center of the room
                else {
                    //average W, E, N, and S from last time step
                    room[EL_INDEX] =
                            (room[PREV_EL_INDEX - 1] +
                             room[PREV_EL_INDEX + 1] +
                             room[PREV_EL_INDEX - X_SIZE] +
                             room[PREV_EL_INDEX + X_SIZE]
                            ) / 4;
                }
            }
        }
        cout << room[((k%2)* TILES) + (LOC_Y_INDEX * X_SIZE) + LOC_X_INDEX] << endl;
    }
    //printArray(room, TILES, X_SIZE, Y_SIZE);
    //printRoomTimeStep(room, TILES, X_SIZE, Y_SIZE, TIME_STEPS);
    const int CP_INDEX = (((TIME_STEPS + 1) % 2) * (TILES));
    return room[CP_INDEX + (LOC_Y_INDEX * X_SIZE) + LOC_X_INDEX];
}


/**********************************************************************
 * This helper function prints the entire room state in a "human-
 * readable format".
 *
 * @param room is the pointer to the room model's temperature state.
 * @param TILES is the number of pieces the room is divided into
 * for temperature computations.
 * @param X_SIZE is the width of the room.
 * @param Y_SIZE is the depth of the room.
 *********************************************************************/
void printArray(const float *room, const int TILES, const int X_SIZE,
                const int Y_SIZE) {
    for (int i = 0; i < TILES * 2; i++) {
        if (i == TILES)
            cout << endl << endl << endl;
        if (i % X_SIZE == 0)
            cout << endl;
        cout << room[i] << " ";
    }
    cout << endl << endl << endl;
}


/**********************************************************************
 * This helper function prints half of the room state in a "human-
 * readable format" (only one time step is printed).
 *
 * @param room is the pointer to the room model's temperature state.
 * @param TILES is the number of pieces the room is divided into
 * for temperature computations.
 * @param X_SIZE is the width of the room.
 * @param Y_SIZE is the depth of the room.
 * @param TIME_STEPS is the time step to display.
 *********************************************************************/
void printRoomTimeStep(const float *room, const int TILES,
                       const int X_SIZE, const int Y_SIZE,
                       int TIME_STEPS) {
    for (int i = 0; i < TILES; i++) {
        if (i % X_SIZE == 0 && i != 0)
            cout << endl;
        cout << room[(((TIME_STEPS + 1) % 2) * TILES) + i] << " ";
    }
    cout << endl;
}


/**********************************************************************
 * This helper function prints half of the rod state in a "human-
 * readable format" (only one time step is printed).
 *
 * @param rod is the pointer to the rod model's temperature state.
 * @param SLICES is the number of pieces the rod is divided into
 * for temperature computations.
 * @param X_SIZE is the width of the room.
 * @param TIME_STEP is the time step to display.
 *********************************************************************/
void printRodTimeStep(const float *rod, const int SLICES,
                      const int X_SIZE, int TIME_STEP) {
    for (int i = 0; i < SLICES; i++) {
        cout << rod[((TIME_STEP % 2) * SLICES) + i] << " ";
    }
    cout << endl;
}