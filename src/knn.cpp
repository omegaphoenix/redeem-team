/*
 * We are using USERS
 */

#include "knn.hpp"
#include <iostream>
#include <time.h>
#include <math.h>

kNN::~kNN() {
}

void kNN::train(std::string saveFile) {
    // Do something with saveFile
    std::cout << "entered train()" << std::endl;
    buildMatrix();
}

// Calculates Pearson correlation for each item
// Optimized to not go through entire data set twice
float kNN::pearson(int i_start, int i_end, int j_start, int j_end) {
    float x_i_ave = 0;
    float x_j_ave = 0;
    int L = 0;
    std::vector<int> ratings_i;
    std::vector<int> ratings_j;

    int i = i_start;
    int j = j_start;

    int movie_i;
    int movie_j;

    while (i < i_end && j < j_end) {
        movie_i = columns[i];
        movie_j = columns[j];
        if (movie_i == movie_j) {
            ratings_i.push_back(values[i]);
            ratings_j.push_back(values[j]);
            x_i_ave += values[i];
            x_j_ave += values[j];
            L++;
            i++;
            j++;
        }
        else if (movie_i < movie_j) {
            i++;
        }
        else {
            j++;
        }
    }

    if (L == 0) { // no movies in common
        return 0;
    }

    // std::cout << "pearson(): nonzero correlation found" << std::endl;

    x_i_ave = x_i_ave / L;
    x_j_ave = x_j_ave / L;

    float nominator = 0;

    float denom1 = 0;
    float denom2 = 0;

    float ith_part;
    float jth_part;

    for (int k = 0; k < ratings_i.size(); k++) {
        ith_part = ratings_i[k] - x_i_ave;
        jth_part = ratings_j[k] - x_j_ave;

        nominator += (ith_part * jth_part);

        denom1 = denom1 + (ith_part * ith_part);
        denom2 = denom2 + (jth_part * jth_part);
    }
    float corr = nominator / sqrt(denom1 * denom2);

    return corr;
}

void kNN::buildMatrix() {
    std::cout << "entered buildMatrix()\n";
    int N = N_USERS;

    // Initialize corrMatrix with three rows
    // 0: correlation
    // 1: user i
    // 2: user j
    // where i < j
    for (int i = 0; i < 3; i++) {
        std::vector<float> row(N_USERS); // is this valid initialization?
        corrMatrix.push_back(row);
    }

    int num_correlations = 0;

    std::cout << "matrix initialized\n";
    for (int i = 0; i < N - 1; i++) {
        for (int j = i; j < N; j++) {
            //std::cout << "Running pearson on " << i << ", " << j << std::endl;
            //corrMatrix[i][j] = pearson(train[i], train[j]);
            float corr = pearson(rowIndex[i], rowIndex[i + 1], rowIndex[j], rowIndex[j + 1]);
            if (corr != 0) {
                corrMatrix[0].push_back(corr);
                corrMatrix[1].push_back(i);
                corrMatrix[2].push_back(j);
                num_correlations++;
                if (num_correlations % 1000000 == 0) {
                    std::cout << "num_correlations = " << num_correlations << std::endl;
                }
            }
        }
    }

    std::cout << "Total number of correlations: " << num_correlations << std::endl;
    return;
}

// Find "closest" movies and average user's ratings for them
void kNN::predict(int user, int movie) {

}

void kNN::loadSaved(std::string fname) {
    loadCSR(fname);
}

int main(int argc, char **argv) {
    clock_t time0 = clock();
    kNN* knn = new kNN();
    clock_t time1 = clock();

    // Load data from file.
    knn->load("1.dta");
    clock_t time2 = clock();

    // Train by building correlation matrix
    std::cout << "Begin training\n";
    knn->train("unused variable");

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