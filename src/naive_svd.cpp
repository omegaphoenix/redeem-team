#include "naive_svd.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <time.h>
#include <random> // For assigning random values
#include <algorithm> // For shuffling
#include <stdlib.h> // For abs()

bool DEBUG = true;

// Clean up U, V.
NaiveSVD::~NaiveSVD() {
    delete this->U;
    delete this->V;
}

void NaiveSVD::setParams(int K, float eta, float lambda) {
    this->K = K;
    this->eta = eta;
    this->lambda = lambda;

    this->MAX_EPOCHS = 30;
    this->EPSILON = 0.0001;
}

// Generic SGD training algorithm.
void NaiveSVD::train() {
    // Initialize U, V
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

    // Get initial error calculation (by calling runEpoch)
    float delta0 = runEpoch();
    int num_epochs = 1;

    // If num epochs left < max_epochs
    while (num_epochs < this->MAX_EPOCHS) {
        // Run an epoch and get the error back
        float delta = runEpoch();
        num_epochs++;
            
        // If the difference in error is less than epsilon, break
        float delta_error = delta / delta0;
        if (DEBUG) {
            std::cout << "ratio of curr_error / init_error is " << delta_error << std::endl;
        }
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
    if (DEBUG) {
        std::cout << "error was " << new_error << std::endl;
    } 

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

    return error;
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


void NaiveSVD::save(std::string fname) {
}

void NaiveSVD::loadSaved(std::string fname) {
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

int main(int argc, char **argv) {
    // Speed up stdio operations
    std::ios_base::sync_with_stdio(false);

    clock_t time0 = clock();
    NaiveSVD* nsvd = new NaiveSVD();
    clock_t time1 = clock();

    // Load in COO format into ratings vector
    nsvd->load("1.dta");
    clock_t time2 = clock();

    std::cout << "Setting parameters" << std::endl;
    clock_t time3 = clock();
    nsvd->setParams(10, 0.001, 0.0);
    clock_t time4 = clock();

    std::cout << "Begin training" << std::endl;
    clock_t time5 = clock();
    nsvd->train();
    clock_t time6 = clock();

    std::cout << "Printing output" << std::endl;
    nsvd->printOutput("out/naive_svd.dta");
    clock_t time7 = clock();

    double ms1 = diffclock(time1, time0);
    std::cout << "Initializing took " << ms1 << " ms" << std::endl;
    double ms2 = diffclock(time2, time1);
    std::cout << "Total loading took " << ms2 << " ms" << std::endl;
    double ms3 = diffclock(time4, time3);
    std::cout << "Setting params took " << ms3 << std::endl;;
    double ms4 = diffclock(time6, time5);
    std::cout << "Training took " << ms4 << std::endl;
    double ms5 = diffclock(time7, time6);
    std::cout << "Printing took " << ms5 << std::endl;
    double total_ms = diffclock(time6, time0);
    std::cout << "Total running time was " << total_ms << " ms" << std::endl;
    return 0;
}
