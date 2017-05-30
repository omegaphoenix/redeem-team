#include "svd_plusplus.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <random> // For assigning random values
#include <algorithm> // For shuffling
#include <stdlib.h> // For abs()
#include <math.h> // For sqrt(), isnan()
#include <assert.h>

#define DEBUG true

SVDPlus::SVDPlus() : Model() {
    validation_loaded = false;
    U = NULL;
    V = NULL;

    // Create a randomly shuffled order of entries to go through.
    /* We won't use a shuffler because we can't save on
    // y, c, w calculations with it.
    std::vector<int> shuffler;
    for (int i = 0; i < this->numRatings; i++) {
        shuffler.push_back(i);
    }
    */
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

    this->MAX_EPOCHS = 30;
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
            save("model/svd_plus/k=" + std::to_string(K) + "_lamb=" +
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
    #ifdef DEBUG
        std::cout << "Computing initial error" << std::endl;
    #endif
    float init_error = computeAllError(); 

    //std::shuffle(this->shuffler.begin(), this->shuffler.end(),
    //    std::default_random_engine(0));

    // To minimize the amount of y, c, w updating operations,
    // we only update them once per unique user per epoch.
    int curr_user = -1;
    bool update_ycw = false;
    // Initialize user-specific variables.
    float N;
    float sum_w;
    float sum_c;
    float* sum_y = new float[this->K];

    // For each data point in the set
    for (int i = 0; i < this->numRatings; i++) {
        int idx = i;
        //int idx = this->shuffler[i];

        // Get the user, movie, rating
        int user = this->ratings[idx * DATA_POINT_SIZE + USER_IDX];
        int movie = this->ratings[idx * DATA_POINT_SIZE + MOVIE_IDX];
        float rating = (float) this->ratings[idx * DATA_POINT_SIZE + RATING_IDX];

        // Check if we need to recalculate user-specific variables.
        if (user != curr_user || curr_user == -1) {
            curr_user = user;

            // Reset user-dependent variables.
            N = 0.0;
            sum_w = 0.0;
            sum_c = 0.0;
            std::fill(sum_y, sum_y + (this->K), 0.0);

            // Fill them with new values.
            if ((this->rowIndex[user] - this->rowIndex[user - 1]) > 0.0) {
                N = 1.0 / std::sqrt((float) (rowIndex[user] - rowIndex[user - 1]));
            }

            getPlusVariables(user, movie, N, sum_y, sum_w, sum_c);

            update_ycw = true;
        }

        // Calculate (rating - est_rating) given user-specific vars.
        float e_ui = rating - predictRating(user, movie, sum_y, 
            sum_w, sum_c);

        // Update the corresponding rows in the U, V matrix
        update(user, movie, rating, N, sum_y, e_ui, update_ycw);
    }
    // Compute the new error.
    float new_error = computeAllError();
    float delta_error = std::abs(new_error - init_error);
    float RMSE = std::sqrt(new_error);
    #ifdef DEBUG
        std::cout << "RMSE was " << RMSE << std::endl;
    #endif

    delete sum_y;

    // Return the error
    return delta_error;

}

// Computes one update step in SGD.
void SVDPlus::update(int user, int movie, float rating, 
    float N, float* sum_y, float e_ui, bool update_ycw) {

    if (update_ycw) {
        // Update user-specific variables. We will do this only
        // once per user (to save time).

        int start = rowIndex[user - 1];
        int end = rowIndex[user];

        // Iterate through all movies a user has rated.
        for (int j = start; j < end; j++) {

            // Well, for now we're only going to iterate 
            // through 10.
            if (j - start > 10) {
                break;
            }
            
            // Get the ID of the movie the user rated.
            int movie_idx = (int) this->columns[j];

            // Update the k-vector of y_j.
            for (int k = 0; k < this->K; k++) {

                float y_jk = this->Y[movie_idx * this->K + k];
                float dy = (this->lambda * y_jk) 
                    - (this->V[movie * this->K + k] * N * e_ui);

                this->Y[movie_idx * this->K + k] -= this->eta * dy;

            }

            // Updating w
            /*
            float w_ij = this->W[movie * N_MOVIES + movie_idx];
            float rating_j = (float) this->values[j];
            float dw = (this->lambda * w_ij)
                - (N * e_ui * (rating_j - getBias(user, movie_idx)));

            this->W[movie * N_MOVIES + movie_idx] -= this->eta * dw;

            // Updating c
            float dc = (this->lambda * this->C[movie * N_MOVIES + movie_idx])
                - (N * e_ui);
            this->C[movie * N_MOVIES + movie_idx] -= this->eta * dc;
            */
        }
    }

    for (int i = 0; i < this->K; i++) {
        // Multiply (Y - (U dot V)) with v_i
        // and subtract from (lambda * u_i) = du
        float P_uk = this->U[user * this->K + i];
        float Q_ik = this->V[movie * this->K + i];
        float du = (this->lambda * P_uk) - (Q_ik * e_ui);

        // Multiply (Y - (U dot V)) with u_i
        // and subtract from (lambda * v_i) = dv
        float dv = (this->lambda * Q_ik) - ((P_uk + sum_y[i]) * e_ui);

        // Set u_i = u_i - (ETA * du)
        this->U[user * this->K + i] -= this->eta * du;

        // Set v_i = v_i - (ETA * dv)
        this->V[movie * this->K + i] -= this->eta * dv;
    }
    /*
    this->user_bias[user] += this->eta * (e_ui - 
        (this->lambda * this->user_bias[user]));
    this->movie_bias[movie] += this->eta * (e_ui - 
        (this->lambda * this->movie_bias[movie]));
    */
}

void SVDPlus::getPlusVariables(int user, int movie, float N, 
    float* sum_y, float &sum_w, float &sum_c) {
    for (int j = rowIndex[user - 1]; j < (rowIndex[user - 1] + 10); j++) {
        int movie_j = (int) this->columns[j];
        float rating = (float) this->values[j];

        // Internal loop for populating the k-vector y_j
        for (int k = 0; k < this->K; k++) {
            float y_jk = this->Y[movie_j * this->K + k];
            sum_y[k] += y_jk * N;
        }

        /*
        // Get |R|^(-1/2) * (rating - b_uj) * w_ij
        // and add it to "nearest neighbors" sum.
        float b_uj = getBias(user, movie_j);
        sum_w += N * (rating - b_uj) * this->W[movie * N_MOVIES + movie_j];

        // Get |N|^(-1/2) * c_ij
        // and add it to "implicit_neighbors" sum
        sum_c += N * this->C[movie * N_MOVIES + movie_j];
        */
    }
}

float SVDPlus::getBias(int user, int movie) {
    return this->user_bias[user] + this->movie_bias[movie] - this->mu;
}

float SVDPlus::predictRating(int user, int movie, float* sum_y, 
    float sum_w, float sum_c) {

    // Get mu + b_u + b_i
    float b_ui = getBias(user, movie);

    // Get (V dot U) + (V dot (N * SUM[y]))
    float QP = dotProduct(user, movie);
    float QY = 0.0;
    for (int i = 0; i < this->K; i++) {
        QY += this->V[movie * this->K + i] * sum_y[i];
    }

    // Finally, calculate the huge hecking thing.
    assert(sum_w == 0.0);
    assert(sum_c == 0.0);
    double result = b_ui + QP + QY;
    // + sum_w + sum_c;
    return result;
}

float SVDPlus::computeAllError() {
    // Initialize error = 0
    float error = 0.0;
    int curr_user = -1;
    float* sum_y = new float[this->K];
    float N;
    float sum_w;
    float sum_c;

    // For all data points,
    for (int i = 0; i < this->numRatings; i++) {

        // Get user, movie, rating
        int user = this->ratings[i * DATA_POINT_SIZE + USER_IDX];
        int movie = this->ratings[i * DATA_POINT_SIZE + MOVIE_IDX];
        float rating = (float) this->ratings[i * DATA_POINT_SIZE + RATING_IDX];

        // Check if the user has changed.
        if (user != curr_user || curr_user == -1) {
            curr_user = user;

            // Reset user-dependent variables.
            N = 0.0;
            sum_w = 0.0;
            sum_c = 0.0;
            std::fill(sum_y, sum_y + (this->K), 0);

            // Fill them with new values.
            if ((this->rowIndex[user] - this->rowIndex[user - 1]) > 0.0) {
                N = 1.0 / std::sqrt((float) (rowIndex[user] - rowIndex[user - 1]));
            }

            getPlusVariables(user, movie, N, sum_y, sum_w, sum_c);
        }

        // Calculate the dot product
        // of U[i] and V[i]
        float e_ui = rating - predictRating(user, movie, sum_y, 
            sum_w, sum_c);

        // Subtract from Y[i] **2
        // Add result to error.
        error += e_ui * e_ui;

    }

    delete sum_y;
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
    std::cout << "SVDPlus::save called" << std::endl;
    FILE *out = fopen(fname.c_str(), "wb");
    if (out == NULL) {
        std::cout << "File doesn't exist" << std::endl;
    }
    int buf[1];
    buf[0] = numEpochs;

    fwrite(buf, sizeof(int), 1, out);
    fwrite(U, sizeof(float), N_USERS * K, out);
    fwrite(V, sizeof(float), N_MOVIES * K, out);

    fwrite(Y, sizeof(float), N_MOVIES * K, out);
    fwrite(C, sizeof(float), N_MOVIES * N_MOVIES, out);
    fwrite(W, sizeof(float), N_MOVIES * N_MOVIES, out);

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
        std::cout << "LOADING FROM FRESH" << std::endl;
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

        fread(Y, sizeof(float), N_MOVIES * K, in);
        fread(C, sizeof(float), N_MOVIES * N_MOVIES, in);
        fread(W, sizeof(float), N_MOVIES * N_MOVIES, in);

        fclose(in);
    }
}

void SVDPlus::printOutput(std::string fname) {
    std::ofstream outputFile;
    outputFile << std::setprecision(3);
    outputFile.open(fname);
    SVDPlus* qual = new SVDPlus();
    qual->load("5-1.dta");

    int curr_user = -1;
    float N;
    float sum_w;
    float sum_c;
    float* sum_y = new float[this->K];

    for (int i = 0; i < qual->numRatings; i++) {
        int user = qual->ratings[i * DATA_POINT_SIZE + USER_IDX];
        int movie = qual->ratings[i * DATA_POINT_SIZE + MOVIE_IDX];

        // Check if we need to recalculate user-specific variables.
        if (user != curr_user || curr_user == -1) {
            curr_user = user;

            // Reset user-dependent variables.
            N = 0.0;
            sum_w = 0.0;
            sum_c = 0.0;
            std::fill(sum_y, sum_y + (this->K), 0);

            // Fill them with new values.
            if ((this->rowIndex[user] - this->rowIndex[user - 1]) > 0.0) {
                N = 1.0 / std::sqrt((float) (rowIndex[user] - rowIndex[user - 1]));
            }

            getPlusVariables(user, movie, N, sum_y, sum_w, sum_c);
        }

        float rating = predictRating(user, movie, sum_y, sum_w, sum_c);
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
    delete sum_y;
}
