#include "naive_svd.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <random> // For assigning random values
#include <algorithm> // For shuffling
#include <stdlib.h> // For abs()
#include <math.h> // For sqrt()

#define DEBUG true

SVDPlus::SVDPlus() : Model() {
    validation_loaded = false;
    U = NULL;
    V = NULL;
}

// Clean up U, V.
SVDPlus::~SVDPlus() {
    delete this->U;
    delete this->V;
}

void SVDPlus::setParams(int K, float eta, float lambda, 
    float mu, float* user_bias, float* movie_bias) {
    this->K = K;
    this->eta = eta;
    this->lambda = lambda;

    this->MAX_EPOCHS = 120;
    this->EPSILON = 0.0001;

    // Need to run a baseline model
    // and get these values before setting up
    // an SVDPlus.
    this->mu = mu;
    this->user_bias = user_bias;
    this->movie_bias = movie_bias;
}

// Generic SGD training algorithm.
void SVDPlus::train(std::string saveFile) {
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
float SVDPlus::runEpoch() {
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
void SVDPlus::update(int user, int movie, float rating) {

    // Calculate error between predicted and actual rating
    e_ui = rating - predictRating(user, movie, rating)

    // Quickly calculate the number of movies a user has rated.
    float N = (float) this->rowIndex[user + 1] - this->rowIndex[user];
    N = 1.0 / std::sqrt(N);

    // Calculate SUM y_j
    float sum_y[this->K] = {};
    for (int j = rowIndex[user]; j < rowIndex[user + 1]; j++) {
        int movie_idx = this->columns[j];

        for (int k = 0; k < this->K; k++) {
            float y_jk = this->Y[movie_idx * this->K + k];
            // At each iteration, add y_j[i] / |N|^(1/2)
            sum_y[k] += y_jk * N;

            // ^^ That'll be used to calculate the new U, V
            // We are going to update the other variables
            // (y, c, w) here.
            float dy = (this->lambda * y_jk) 
                - (this->V[movie * this->K + k] * N * e_ui);
            this->Y[movie_idx * this->K + k] -= this->eta * dy;
        }
        // Updating w
        float dw = (this->lambda * this->W[movie * N_MOVIES + movie_idx])
            - (N * e_ui * (rating - getBias(user, movie)));
        this->W[movie * N_MOVIES + movie_idx] -= this->eta * dw;

        // Updating c
        float dc = (this->lambda * this->C[movie * N_MOVIES + movie_idx])
            - (N * e_ui);
        this->C[movie * N_MOVIES + movie_idx] -= this->eta * dc;
    }

    for (int i = 0; i < this->K; i++) {
        // Multiply (Y - (U dot V)) with v_i
        // and subtract from (lambda * u_i) = du
        du = (this->lambda * this->U[user * this->K + i])
            - (this->V[movie * this->K + i] * e_ui);

        // Multiply (Y - (U dot V)) with u_i
        // and subtract from (lambda * v_i) = dv
        dv = (this->lambda * this->V[movie * this->K + i])
            - ((this->U[user * this->K + i] + sum_y[i]) 
            * e_ui);

        // Set u_i = u_i - (ETA * du)
        this->U[user * this->K + i] = this->eta * du;

        // Set v_i = v_i - (ETA * dv)
        this->V[movie * this->K + i] = this->eta * dv;
    }

}

float SVDPlus::getBias(int user, int movie) {
    return this->user_bias[user] + this->movie_bias[movie] - this->mu;
}

float SVDPlus::predictRating(int user, int movie, float rating) {
    // Get mu + b_u + b_i
    float b_ui = getBias(user, movie);

    // Get |N|^(-1/2) * SUM[y_j]
    float N = (float) this->rowIndex[user + 1] - this->rowIndex[user];
    N = 1.0 / std::sqrt(N);
    float sum_y[this->K] = {};
    float nearest_neighbors = 0.0;
    float implicit_neighbors = 0.0;

    for (int j = rowIndex[user]; j < rowIndex[user + 1]; j++) {
        int movie_j = this->columns[j];

        for (int k = 0; k < this->K; k++) {
            float y_jk = this->Y[movie_j * this->K + k];
            // At each iteration, add y_j[i] / |N|^(1/2)
            sum_y[k] += y_jk * N;
        }
        // Get |R|^(-1/2) * SUM[rating - b_uj] * w_ij
        b_uj = getBias(user, movie_j);
        nearest_neighbors += N * (rating - b_uj) 
            * this->W[movie * N_MOVIES + movie_j];

        // Get |N|^(-1/2) * SUM[c_ij]
        implicit_neighbors += N * this->C[movie * N_MOVIES + movie_j];

    }

    // Get (V dot U) + (V dot (N * SUM[y]))
    QP = dotProduct(user, movie);
    float QY = 0.0;
    for (int i = 0; i < this->K; i++) {
        QY += this->V[movie * this->K + i] * sum_y[i];
    }

    // Finally, calculate the huge hecking thing.
    float result = b_ui + QP + QY + nearest_neighbors + implicit_neighbors;
    return result
}

float SVDPlus::computeAllError() {
    // Initialize error = 0
    float error = 0.0;

    // For all data points,
    for (int i = 0; i < this->numRatings; i++) {
        int user = this->ratings[i * DATA_POINT_SIZE + USER_IDX];
        int movie = this->ratings[i * DATA_POINT_SIZE + MOVIE_IDX];
        float rating = (float) this->ratings[i * DATA_POINT_SIZE + RATING_IDX];

        // Calculate the dot product
        // of U[i] and V[i]
        float e_ui = rating - predictRating(user, movie);

        // Subtract from Y[i] **2
        // Add result to error.
        error += e_ui * e_ui;

    }

    return error / numRatings;
}

float SVDPlus::dotProduct(int user, int movie) {
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

float SVDPlus::validate(std::string valFile, std::string saveFile) {
    // Load ratings
    if (!validation_loaded) {
        load(valFile);
        validation_loaded = true;
    }
    // Load U, V
    loadSaved(saveFile);

    return computeAllError();
}

// Use <stdio.h> for binary writing.
void SVDPlus::save(std::string fname) {
    FILE *out = fopen(fname.c_str(), "wb");
    int buf[1];
    buf[0] = numEpochs;
    fwrite(buf, sizeof(int), 1, out);
    fwrite(U, sizeof(float), N_USERS * K, out);
    fwrite(V, sizeof(float), N_MOVIES * K, out);

    fwrite(Y, sizeof(float), N_MOVIES * K, out);
    fwrite(C, sizeof(float), N_MOVIES * N_MOVIES, out);
    fwrite(W, sizeof(float), N_MOVIES * N_MOVIES, out);

    /* Not sure if we're doing these here.
    fwrite(mu, sizeof(float), 1, out);
    fwrite(user_bias, sizeof(float), N_USERS, out);
    fwrite(movie_bias, sizeof(float), N_MOVIES, out);
    */

    fclose(out);
}

// Initialize U, V from saved file or randomly
// between [-0.5, 0.5] if the saved file failed
// to load.
void SVDPlus::loadSaved(std::string fname) {
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
        // Initialize the W, C, Y arrays.
        this->Y = new float[N_MOVIES * this->K];
        this->C = new float[N_MOVIES * N_MOVIES];
        this->W = new float[N_MOVIES * N_MOVIES];

        // Initialize values to be between -0.5, 0.5
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(-0.5, 0.5);
        for (int i = 0; i < N_USERS * this->K; i++) {
            *(this->U + i) = distribution(generator);

        }
        for (int i = 0; i < N_MOVIES * this->K; i++) {
            *(this->V + i) = distribution(generator);
            *(this->Y + i) = distribution(generator);
        }
        for (int i = 0; i < N_MOVIES * N_MOVIES; i++) {
            *(this->C + i) = distribution(generator);
            *(this->W + i) = distribution(generator);
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

        // Initialize the W, C, Y arrays.
        this->Y = new float[N_MOVIES * this->K];
        this->C = new float[N_MOVIES * N_MOVIES];
        this->W = new float[N_MOVIES * N_MOVIES];

        fread(Y, sizeof(float), N_MOVIES * K, out);
        fread(C, sizeof(float), N_MOVIES * N_MOVIES, out);
        fread(W, sizeof(float), N_MOVIES * N_MOVIES, out);

        fclose(in);
    }
}

void SVDPlus::printOutput(std::string fname) {
    std::ofstream outputFile;
    outputFile << std::setprecision(3);
    outputFile.open(fname);
    SVDPlus* qual = new SVDPlus();
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
