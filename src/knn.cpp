/*
 * We are using USERS
 */

#include "knn.hpp"
#include <iostream>
#include <time.h>
#include <math.h>

kNN::~kNN() {
}

void kNN::train() {
    std::cout << "entered train()" << std::endl;
    std::cout << "ratings size = " << ratings.size() << std::endl;
    buildMatrix(ratings);
}

// Calculates Pearson correlation for each item
// Optimized to not go through entire data set twice
float kNN::pearson(std::vector<float> x_i, std::vector<float> x_j) {
    float x_i_ave = 0;
    float x_j_ave = 0;
    int L = 0;
    std::vector<int> index;

    for (int i = 0; i < x_i.size(); i++) {
        if (x_i[i] != 0 && x_j[i] != 0) {
            x_i_ave = x_i_ave + x_i[i];
            x_j_ave = x_j_ave + x_j[i];
            L++;
            // Storing index.  Don't want to go through entire matrix twice
            index.push_back(i);
        }
    }

    if (L == 0) {
        return 0;
    }

    x_i_ave = x_i_ave / L;
    x_j_ave = x_j_ave / L;

    float nominator = 0;

    float denom1 = 0;
    float denom2 = 0;

    float ith_part;
    float jth_part;

    for (int i = 0; i < index.size(); i++) {
        ith_part = x_i[index[i]] - x_i_ave;
        jth_part = x_j[index[i]] - x_j_ave;

        nominator = nominator + (ith_part * jth_part);

        denom1 = denom1 + (ith_part * ith_part);
        denom2 = denom2 + (jth_part * jth_part);
    }
    float corr = nominator/sqrt(denom1 * denom2);

    return corr;
}

void kNN::buildMatrix(std::vector<std::vector<float>> &train) {
    std::cout << "entered buildMatrix()\n";
    int N = N_USERS;

    // NEED TO INIT corrMatrix
    //corrMatrix.clear();
    //std::cout << "matrix cleared\n";
    //corrMatrix.resize(N_USERS, std::vector<float>(N_USERS, 0));

    // for (int i = 0; i < N_USERS; i++) {
    //     std::vector<float> row;
    //     for (int j = 0; j < N_USERS; j++) {
    //         row.push_back(0);
    //     }
    //     std::cout << "push row " << i << std::endl;
    //     corrMatrix.push_back(row);
    // }

    // Initialize corrMatrix with three rows
    // 0: correlation
    // 1: user i
    // 2: user j
    // where i < j
    for (int i = 0; i < 3; i++) {
        std::vector<float> row;
        corrMatrix.push_back(row);
    }


    std::cout << "matrix initialized\n";
    for (int i = 0; i < N - 1; i++) {
        for (int j = 0; j < N; j++) {
            //std::cout << "Running pearson on " << i << ", " << j << std::endl;
            //corrMatrix[i][j] = pearson(train[i], train[j]);
            float corr = pearson(train[i], train[j]);
            if (corr != 0) {
                corrMatrix[0].push_back(corr);
                corrMatrix[1].push_back(i);
                corrMatrix[2].push_back(j);
                std::cout << "pushed a nonzero correlation";
            }
        }
    }

    return;
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
    std::cout << "Begin training\n";
    knn->train();

    clock_t time3 = clock();

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

    double kNN_ms = diffclock(time3, time0);
    std::cout << "kNN took " << kNN_ms << " ms" << std::endl;

    return 0;
}