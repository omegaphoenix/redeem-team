/*
 * We are using USERS
 */

#include "knn.hpp"
#include <iostream>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <queue>

kNN::~kNN() {
}

void kNN::train(std::string saveFile) {
    std::cout << "entered train()" << std::endl;
    struct stat buffer;
    if (stat(saveFile.c_str(), &buffer) == 0) {
        std::cout << "File exists. Loading file...\n";
        loadSaved(saveFile);
    }
    else {
        std::cout << "File does not exist. Building matrix...\n";
        buildMatrix(saveFile);
    }
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

    if (L <= shared_threshold) { // ignore user pairs with little in common
        return 0;
    }

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

void kNN::buildMatrix(std::string saveFile) {
    std::cout << "entered buildMatrix()\n";

    // Number of items
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
            if (rowIndex[i + 1] - rowIndex[i] < individual_threshold
                || rowIndex[j + 1] - rowIndex[j] < individual_threshold) {
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

    // Save correlation matrix
    std::cout << "Saving model..." << std::endl;
    save(saveFile);
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

// Find "closest" users and average their ratings of given movie
void kNN::predict(int user, int movie) {
    std::priority_queue<float> corr;
    int count = 0;
    // put correlations in a priority queue
    for (int i = 0; i < num_correlations; i++) {
        float c = corrMatrix[0][i];
        if (c == 1) {
            count++;
        }
        if (!isnan(c)) {
            corr.push(c);
        }
    }
    std::cout << "number of 1s = " << count << std::endl;

    // get top K's average
    int K = 10;
    float avg = 0;
    for (int i = 0; i < K; i++) {
        if (corr.empty()) {
            break;
        }
        // avg += corr.pop();
        float p = corr.top();
        corr.pop();
        std::cout << i << " top corr value = " << p << std::endl;
    }

    // return avg / K;
}

void kNN::loadSaved(std::string fname) {
    FILE *in = fopen(fname.c_str(), "r");

    // Read num correlations
    int buf[1];
    fread(buf, sizeof(int), 1, in);
    num_correlations = buf[0];

    // Initialize correlation matrix
    for (int i = 0; i < 3; i++) {
        std::vector<float> row(num_correlations, 0);
        corrMatrix.push_back(row);
    }

    // Read matrix
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

    std::string data_file = "4.dta";

    // Load data from file.
    knn->load(data_file);

    // Train by building correlation matrix
    knn->metric = kPearson;
    knn->shared_threshold = 2;
    knn->individual_threshold = 7;
    std::cout << "Begin training\n";

    // std::string corr_file = "knn_pearson_s" + std::to_string(shared_threshold)
    //     + "_i" + std::to_string(individual_threshold)
    //     + ".save";

    knn->train("test_knn_corrMatrix.save");

    clock_t time1 = clock();

    // Predict ratings
    // Load qual data
    //knn->load("1.dta");
    knn->predict(1, 1);

    clock_t time2 = clock();

    // Write predictions to file

    // Output times.
    double kNN_ms = diffclock(time1, time0);
    std::cout << "kNN training took " << kNN_ms << " ms" << std::endl;
    double predict_ms = diffclock(time2, time1);
    std::cout << "kNN prediction took " << kNN_ms << " ms" << std::endl;

    return 0;
}