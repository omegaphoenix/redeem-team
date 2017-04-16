/*
 * We are using USERS
 */

#include "knn.hpp"
#include <iostream>
#include <time.h>

kNN::~kNN() {
}

void kNN::train() {
    buildMatrix(ratings, false);
    return;
}

void kNN::pearson(float *x_i, float *x_j) {

}

void kNN::buildMatrix(std::vector<std::vector<float>> train, bool movie) {
    // int N = N_MOVIES;
    // if (!movie) {
    //     N = N_USERS;
    // }

    // for (int i = 0; i < N - 1; i++) {
    //     for (int j = 0; j < N; j++) {
    //         if (!movie) {
    //             corrMatrix[i][j] = pearson(train[i], train[j]);
    //         }
    //         else {
    //             // shouldn't get here; figure out how to optimize
    //             std::cout << "cannot build matrix based on movies\n";
    //         }
    //     }
    // }

    // return;
}

// Find "closest" movies and average user's ratings for them
void kNN::predict(int user, int movie) {

}

// Returns the differences in ms.
static double diffclock(clock_t clock1, clock_t clock2) {
  double diffticks = clock1 - clock2;
  double diffms = (diffticks) / (CLOCKS_PER_SEC / 1000);
  return diffms;
}

int main(int argc, char **argv) {
    clock_t time0 = clock();
    kNN* knn = new kNN();
    clock_t time1 = clock();

    // Load data from file.
    knn->loadFresh("data/um/1.dta");
    clock_t time2 = clock();

    // Train by building correlation matrix
    knn->train();

    // Predict ratings
    // Load qual data
    knn->predict(0, 0);

    // Write predictions to file

    // Output times.
    double ms1 = diffclock(time1, time0);
    std::cout << "Initialization took " << ms1 << " ms" << std::endl;
    double ms2 = diffclock(time2, time1);
    std::cout << "Loading took " << ms2 << " ms" << std::endl;
    double total_ms = diffclock(time2, time0);
    std::cout << "Total took " << total_ms << " ms" << std::endl;

    return 0;
}