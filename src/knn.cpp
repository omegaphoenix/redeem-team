/*
 * We are using USERS
 */
#include "baseline.hpp"
#include "knn.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <queue>

/* Stores correlation value and correlated user. */
struct corrUser {
    float corr;
    int user;

    corrUser(float c, int u) : corr(c), user(u) {
    }

    bool operator<(const struct corrUser &other) const
    {
        return corr < other.corr;
    }
};

kNN::kNN() : Model() {
    this->normalized_values = new float[N_TRAINING];
}

kNN::~kNN() {
    delete this->normalized_values;
}

void kNN::normalizeRatings(float average_array[], float stdev_array[]) {
    int user = 0;
    for (int i = 0; i < N_TRAINING; i++) {
        if (i == rowIndex[user + 1]) {
            user++;
        }
        normalized_values[i] = (values[i] - average_array[user]) / stdev_array[user];
    }
}

void kNN::train(std::string saveFile) {
    std::cout << "Begin training..." << std::endl;
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
        corrMatrix.push_back(row);
    }

    num_correlations = 0;
    clock_t prev_clock = clock();
    clock_t curr_clock;
    int curr_user = 0;

    std::cout << "matrix initialized\n";

    for (int i = 0; i < N - 1; i++) {
        for (int j = i + 1; j < N; j++) {
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
float kNN::predict(int user, int movie) {
    std::priority_queue<corrUser> top_corr;
    // int count = 0;
    // put correlations in a priority queue
    for (int i = 0; i < num_correlations; i++) {
        float c = corrMatrix[0][i];
        // if (c == 1) {
        //     count++;
        // }
        if (!isnan(c)) {
            if (corrMatrix[1][i] == user) {
                top_corr.push(corrUser(c, corrMatrix[2][i]));
            }
            else if (corrMatrix[2][i] == user) {
                top_corr.push(corrUser(c, corrMatrix[1][i]));;
            }
        }
    }
    //std::cout << "number of 1s = " << count << std::endl;

    // get top K's average
    int K = 10;
    float total = 0;
    int actualK = 0;
    while (actualK < K) {
        if (top_corr.empty()) {
            break;
        }
        // avg += corr.pop();
        corrUser top = top_corr.top();
        top_corr.pop();

        // ignore negative correlations - remove after implementing weighted avg
        if (top.corr < 0) {
            break;
        }

        // std::cout << actualK << " top corr value = " << top.corr << " with user " << top.user << std::endl;

        // check if correlated user has rated movie
        int start_index = rowIndex[top.user];
        int end_index = rowIndex[top.user + 1];
        for (int i = start_index; i < end_index; ++i) {
            if (columns[i] == movie) {
                // TODO: implement weighted average of rankings
                total += values[i];
                ++actualK;
            }
            else if (columns[i] > movie) {
                break;
            }
        }
    }

    if (actualK <= 0) {
        return baseline;
    }

    // for debugging purposes only
    if (total != 0) {
        std::cout << "total = " << total << "\n";
        std::cout << "actualK = " << actualK << "\n";
        std::cout << "got a value!! rating = " << total / actualK << std::endl;
    }

    return total / actualK;
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

std::string kNN::getFilename(std::string data_file) {
    std::string metric_name = "";
    if (metric == kPearson) {
        metric_name = "pearson";
    }
    else if (metric == kSpearman) {
        metric_name = "spearman";
    }
    else {
        metric_name = "unknown";
    }

    // remove '.' from file name
    data_file.erase(
        std::remove(data_file.begin(), data_file.end(), '.'), data_file.end());

    std::string fname = "knn_" + metric_name
        + "_s" + std::to_string(shared_threshold)
        + "_i" + std::to_string(individual_threshold)
        + "_k" + std::to_string(K)
        + "_train=" + data_file
        + ".save";
    return fname;
}

int main(int argc, char **argv) {
    clock_t time0 = clock();
    kNN* knn = new kNN();
    Baseline* base = new Baseline();


    std::string data_file = "4.dta";

    // Load data from file.
    knn->load(data_file);
    knn->baseline = 3; // TODO: figure out what to use
    base->load(data_file);

    base->train("unused variable");
    
    for (int i = 0; i < 1000 ; i++) {
        std::cout << "average " << base->average_array[i] << std::endl;
        std::cout << "std " << base->stdev_array[i] << std::endl;
    }

    knn->normalizeRatings(base->average_array, base->stdev_array);


    // Train by building correlation matrix
    knn->metric = kPearson;
    knn->shared_threshold = 2;
    knn->individual_threshold = 7;
    knn->K = 10;
    std::cout << "Begin training: ";

    std::cout << knn->getFilename(data_file) << "\n";

    knn->train("bin/knn/" + knn->getFilename(data_file));

    clock_t time1 = clock();

    // Predict ratings and write to file
    // Load qual data
    std::cout << "PREDICTIONS:\n";
    kNN* qual = new kNN();
    qual->load("5-1.dta"); // is this how we're going to do things

    // Prepare file
    std::ofstream outputFile;
    outputFile << std::setprecision(3);
    outputFile.open("out/knn.dta");

    for (int i = 0; i < qual->numRatings; i++) {
        int user = qual->ratings[i * DATA_POINT_SIZE + USER_IDX];
        int movie = qual->ratings[i * DATA_POINT_SIZE + MOVIE_IDX];
        float rating = knn->predict(user, movie);
        if (rating < 1) {
            rating = 1;
        }
        else if (rating > 5) {
            rating = 5;
        }

        outputFile << rating << "\n";

        // std::cout << "user " << user << " will rate movie " << movie << ": " << rating << "\n";
        // if (user % 1000 == 0) {
        //     std::cout << "predicting for user " << user << "\n";
        // }
    }
    outputFile.close();

    clock_t time2 = clock();

    // Output times.
    double kNN_ms = diffclock(time1, time0);
    std::cout << "kNN training took " << kNN_ms << " ms" << std::endl;
    double predict_ms = diffclock(time2, time1);
    std::cout << "kNN prediction took " << predict_ms << " ms" << std::endl;

    return 0;
}