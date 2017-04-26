#include "naive_svd.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <random> // For assigning random values
#include <algorithm> // For shuffling
#include <stdlib.h> // For abs()

#define DEBUG true

NaiveSVD::NaiveSVD() : Model() {
    validation_loaded = false;
    U = NULL;
    V = NULL;
}

// Clean up U, V.
NaiveSVD::~NaiveSVD() {
    delete this->U;
    delete this->V;
}

void NaiveSVD::setParams(int K, float eta, float lambda) {
    this->K = K;
    this->eta = eta;
    this->lambda = lambda;

    this->MAX_EPOCHS = 60;
    this->EPSILON = 0.0001;
}

// Generic SGD training algorithm.
void NaiveSVD::train(std::string saveFile) {
    // Initialize U, V
    loadSaved(saveFile);

    #ifdef DEBUG
        std::cout << "Starting with " << numEpochs <<
                  " epochs completed" << std::endl;
    #endif

    float delta0;
    // Get initial error calculation (by calling runEpoch)
    if (numEpochs < this->MAX_EPOCHS) {
        clock_t time0 = clock();
        delta0 = runEpoch();
        clock_t time1 = clock();
        numEpochs++;
        #ifdef DEBUG
            std::cout << "Finished epoch " << numEpochs << std::endl;
            std::cout << "This took: " << diffclock(time1, time0) << " ms.\n";
        #endif
    }
    // If num epochs left < max_epochs
    while (numEpochs < this->MAX_EPOCHS) {
        // Run an epoch and get the error back
        float delta = runEpoch();
        numEpochs++;
        #ifdef DEBUG
            std::cout << "Finished epoch " << numEpochs << std::endl;
        #endif

        if (numEpochs % 10 == 0 || numEpochs == MAX_EPOCHS) {
            save("model/naive_svd/k=" + std::to_string(K) + "_lamb=" +
                 std::to_string(lambda) + "_epoch=" +
                 std::to_string(numEpochs) + ".save");
        }

        // If the difference in error is less than epsilon, break
        float delta_error = delta / delta0;
        #ifdef DEBUG
            std::cout << "ratio of curr_error / init_error is " << delta_error << std::endl;
        #endif
        if (delta_error < this->EPSILON) {
            break;
        }
    }

}

// Run one epoch of SGD, returning delta error.
float NaiveSVD::runEpoch() {
    // Compute the initial error
    float init_error = computeError();

    std::vector<int> shuffler;
    for (int i = 0; i < this->numRatings; i++) {
        shuffler.push_back(i);
    }
    std::shuffle(shuffler.begin(), shuffler.end(),
        std::default_random_engine(0));

    // For each data point in the set
    for (int i = 0; i < this->numRatings; i++) {
        int idx = shuffler[i];

        // Get the user, movie, rating
        int user = this->ratings[idx * DATA_POINT_SIZE + USER_IDX];
        int movie = this->ratings[idx * DATA_POINT_SIZE + MOVIE_IDX];
        float rating = (float) this->ratings[idx * DATA_POINT_SIZE + RATING_IDX];

        // Update the corresponding rows in the U, V matrix
        update(user, movie, rating);
    }
    // Compute the new error.
    float new_error = computeError();
    float delta_error = std::abs(new_error - init_error);
    #ifdef DEBUG
        std::cout << "error was " << new_error << std::endl;
    #endif

    // Return the error
    return delta_error;

}

// Computes one update step in SGD.
void NaiveSVD::update(int user, int movie, float rating) {

    // Calculate the dot product of U, V
    float u_dot_v = dotProduct(user, movie);

    // Calculate Y - (U dot V)
    float intermediate = rating - u_dot_v;

    // Initialize du, dv
    float du;
    float dv;

    for (int i = 0; i < this->K; i++) {
        // Multiply (Y - (U dot V)) with v_i
        // and subtract from (lambda * u_i) = du
        du = (this->lambda * this->U[user * this->K + i])
            - (this->V[movie * this->K + i] * intermediate);

        // Multiply (Y - (U dot V)) with u_i
        // and subtract from (lambda * v_i) = dv
        dv = (this->lambda * this->V[movie * this->K + i])
            - (this->U[user * this->K + i] * intermediate);

        // Set u_i = u_i - (ETA * du)
        this->U[user * this->K + i] -= this->eta * du;

        // Set v_i = v_i - (ETA * dv)
        this->V[movie * this->K + i] -= this->eta * dv;
    }

}

float NaiveSVD::computeError() {
    // Initialize error = 0
    float error = 0.0;

    // For all data points,
    for (int i = 0; i < this->numRatings; i++) {
        int user = this->ratings[i * DATA_POINT_SIZE + USER_IDX];
        int movie = this->ratings[i * DATA_POINT_SIZE + MOVIE_IDX];
        float rating = (float) this->ratings[i * DATA_POINT_SIZE + RATING_IDX];

        // Calculate the dot product
        // of U[i] and V[i]
        float u_dot_v = dotProduct(user, movie);

        // Subtract from Y[i] **2
        // Add result to error.
        error += (rating - u_dot_v) * (rating - u_dot_v);

    }

    return error / numRatings;
}

float NaiveSVD::dotProduct(int user, int movie) {
    // Init result = 0
    float result = 0.0;
    int u_start = user * this->K;
    int v_start = movie * this->K;

    // For i = 0...K-1
    for (int i = 0; i < this->K; i++) {
        result += this->U[u_start + i] * this->V[v_start + i];
    }

    return result;
}

float NaiveSVD::validate(std::string valFile, std::string saveFile) {
    // Load ratings
    if (!validation_loaded) {
        load(valFile);
        validation_loaded = true;
    }
    // Load U, V
    loadSaved(saveFile);

    return computeError();
}

// Use <stdio.h> for binary writing.
void NaiveSVD::save(std::string fname) {
    FILE *out = fopen(fname.c_str(), "wb");
    int buf[1];
    buf[0] = numEpochs;
    fwrite(buf, sizeof(int), 1, out);
    fwrite(U, sizeof(float), N_USERS * K, out);
    fwrite(V, sizeof(float), N_MOVIES * K, out);
    fclose(out);
}

// Initialize U, V from saved file or randomly
// between [-0.5, 0.5] if the saved file failed
// to load.
void NaiveSVD::loadSaved(std::string fname) {
    if (U != NULL) {
        delete U;
    }
    if (V != NULL) {
        delete V;
    }
    FILE *in = fopen(fname.c_str(), "r");
    if (fname == "" || in == NULL) {
        this->U = new float[N_USERS * this->K];
        this->V = new float[N_MOVIES * this->K];

        // Initialize values to be between -0.5, 0.5
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(-0.5, 0.5);
        for (int i = 0; i < N_USERS * this->K; i++) {
            *(this->U + i) = distribution(generator);

        }
        for (int i = 0; i < N_MOVIES * this->K; i++) {
            *(this->V + i) = distribution(generator);
        }
        numEpochs = 0;
    }
    else {
        #ifdef DEBUG
            std::cout << "Loading file: " << fname << std::endl;
        #endif
        // Buffer to hold numEpochs
        int buf[1];
        fread(buf, sizeof(int), 1, in);
        numEpochs = buf[0];

        // Initialize U, V
        this->U = new float[N_USERS * K];
        this->V = new float[N_MOVIES * K];
        fread(U, sizeof(float), N_USERS * K, in);
        fread(V, sizeof(float), N_MOVIES * K, in);
        fclose(in);
    }
}

void NaiveSVD::printOutput(std::string fname) {
    std::ofstream outputFile;
    outputFile << std::setprecision(3);
    outputFile.open(fname);
    NaiveSVD* qual = new NaiveSVD();
    qual->load("5-1.dta");
    for (int i = 0; i < qual->numRatings; i++) {
        int user = qual->ratings[i * DATA_POINT_SIZE + USER_IDX];
        int movie = qual->ratings[i * DATA_POINT_SIZE + MOVIE_IDX];
        float rating = dotProduct(user, movie);
        if (rating < 1) {
            rating = 1;
        }
        else if (rating > 5) {
            rating = 5;
        }
        // Use \n for efficiency.
        outputFile << rating << "\n";
    }
    outputFile.close();
}
