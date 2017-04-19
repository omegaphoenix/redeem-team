/*
 * We are using USERS
 */

#include "knn.hpp"
#include <iostream>
#include <time.h>
#include <math.h>
#include <stdio.h>

kNN::~kNN() {
}

void kNN::train(std::string saveFile) {
    // Do something with saveFile
    std::cout << "entered train()" << std::endl;
    buildMatrix();
}

// Calculates Pearson correlation for each item
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

    if (L <= 2) { // ignore user pairs with little in common
        return 0;
    }

    //std::cout << "number of movies in common: " << L << std::endl;

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
        std::vector<float> row;
        // std::vector<float> row(N_USERS); // is this efficient initialization?
        corrMatrix.push_back(row);
    }

    num_correlations = 0;
    clock_t prev_clock = clock();
    clock_t curr_clock;
    int curr_user = 0;

    std::cout << "matrix initialized\n";

    for (int i = 0; i < N - 1; i++) {
        for (int j = i; j < N; j++) {
            // Don't bother if either user has too few movies
            if (rowIndex[i + 1] - rowIndex[i] < 7
                || rowIndex[j + 1] - rowIndex[j] < 7) {
                if (i % 1000 == 0 && i != curr_user) {
                    curr_user = i;
                    std::cout << "user i = " << i << std::endl;
                }
                continue;
            }

            float corr = pearson(rowIndex[i], rowIndex[i + 1], rowIndex[j], rowIndex[j + 1]);
            if (corr != 0) {
                corrMatrix[0].push_back(corr);
                corrMatrix[1].push_back(i);
                corrMatrix[2].push_back(j);
                num_correlations++;
                if (num_correlations % 1000 == 0) {
                    std::cout << "num_correlations = " << num_correlations << std::endl;
                    std::cout << "current user i = " << i << std::endl;
                }
            }
            // if (i % 100 == 0 && i != curr_user) {
            //     curr_user = i;
            //     curr_clock = clock();
            //     std::cout << "user i = " << i << std::endl;
            //     std::cout << "it took " << diffclock(curr_clock, prev_clock) << " ms\n";
            //     std::cout << "currently we have " << num_correlations << " correlations\n";
            //     prev_clock = clock();
            // }
        }
    }

    std::cout << "Total number of correlations: " << num_correlations << std::endl;

    // Test saving
    std::cout << "Saving model..." << std::endl;
    save("test_knn_corrMatrix.save");
    std::cout << "Saved!" << std::endl;
    std::cout << "SAVED num_correlations = " << num_correlations << std::endl;
    std::cout << "SAVED corrMatrix[0][0] = " << corrMatrix[0][0] << std::endl;

    // Testing loading
    std::cout << "Loading model..." << std::endl;
    loadSaved("test_knn_corrMatrix.save");
    std::cout << "Loaded!" << std::endl;
    std::cout << "LOADED num_correlations = " << num_correlations << std::endl;
    std::cout << "LOADED corrMatrix[0][0] = " << corrMatrix[0][0] << std::endl;

    return;
}

// Find "closest" movies and average user's ratings for them
void kNN::predict(int user, int movie) {

}

void kNN::loadSaved(std::string fname) {
    FILE *in = fopen(fname.c_str(), "r");
    // Check for errors

    // Read num correlations
    int buf[1];
    fread(buf, sizeof(int), 1, in);
    num_correlations = buf[0];

    // Initialize correlation matrix
    fread(&corrMatrix[0][0], sizeof(float), num_correlations, in);
    fread(&corrMatrix[1][0], sizeof(float), num_correlations, in);
    fread(&corrMatrix[2][0], sizeof(float), num_correlations, in);
    fclose(in);
}

void kNN::save(std::string fname) {
    FILE *out = fopen(fname.c_str(), "wb");
    int buf[1];
    buf[0] = num_correlations = corrMatrix[0].size();
    fwrite(buf, sizeof(int), 1, out);

    fwrite(&corrMatrix[0][0], sizeof(float), corrMatrix[0].size(), out);
    fwrite(&corrMatrix[1][0], sizeof(float), corrMatrix[1].size(), out);
    fwrite(&corrMatrix[2][0], sizeof(float), corrMatrix[2].size(), out);
    fclose(out);
}

int main(int argc, char **argv) {
    clock_t time0 = clock();
    kNN* knn = new kNN();

    // Load data from file.
    knn->load("4.dta");

    // Train by building correlation matrix
    std::cout << "Begin training\n";
    knn->train("unused variable");

    clock_t time1 = clock();

    // Predict ratings
    // Load qual data
    knn->predict(0, 0);

    // Write predictions to file

    // Output times.
    double kNN_ms = diffclock(time1, time0);
    std::cout << "kNN took " << kNN_ms << " ms" << std::endl;

    return 0;
}