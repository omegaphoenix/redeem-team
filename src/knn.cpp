#include "baseline.hpp"
#include "knn.hpp"
#include <algorithm>
#include <cmath>
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
    int rating;

    corrUser(float c, int u, int r) : corr(c), user(u), rating(r) {
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

void kNN::normalizeRatings(double average_array[], double stdev_array[]) {
    int user = 0;
    int nan_count = 0;
    int stdev_count = 0;

    for (int i = 0; i < numRatings; i++) {
        while (i == rowIndex[user + 1]) {
            user++;
        }
        normalized_values[i] = (values[i] - average_array[user]) / stdev_array[user];
    }
}

// float kNN::denormalize(float normalized, int user) {
//     return (normalized * stdev_array[user]) + average_array[user];
// }

void kNN::train(std::string saveFile) {
    std::cout << "Begin training: " << saveFile << std::endl;
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
    // 0: index at which user i's correlations begin (int)
    // 1: user j (int)
    // 2: correlation (float)
    for (int i = 0; i < 3; i++) {
        std::vector<float> row;
        corrMatrix.push_back(row);
    }

    num_correlations = 0; // also acts as index
    int curr_user = 0;
    int num_users = 0;

    std::cout << "matrix initialized\n";

    // initialize CSR
    corrMatrix[0].push_back(-1);

    for (int i = 0; i < N; i++) {
        corrMatrix[0].push_back(num_correlations);
        for (int j = i + 1; j < N; j++) {
            // Don't bother if either user has too few movies
            if (rowIndex[i + 1] - rowIndex[i] < individual_threshold
                || rowIndex[j + 1] - rowIndex[j] < individual_threshold) {
                // print for debugging
                if (i % 1000 == 0 && i != curr_user) {
                    curr_user = i;
                    std::cout << "user i = " << i << std::endl;
                }
                continue;
            }
            num_users++;

            float corr = pearson(rowIndex[i], rowIndex[i + 1], rowIndex[j], rowIndex[j + 1]);
            if (corr != 0) {
                corrMatrix[1].push_back(j);
                corrMatrix[2].push_back(corr);
                num_correlations++;
                if (num_correlations % 1000 == 0) {
                    std::cout << "num_correlations = " << num_correlations << std::endl;
                    std::cout << "current user i = " << i << std::endl;
                }
            }
        }
    }
    corrMatrix[0].push_back(num_correlations);

    std::cout << "size of corrMatrix[0] (should be 1 greater than N_USERS) = "
        << corrMatrix[0].size() << "\n";

    std::cout << "Total number of correlations: " << num_correlations << "\n";
    std::cout << "Total number of users above threshold: " << num_users << "\n";

    // Save correlation matrix
    std::cout << "Saving model..." << std::endl;
    save(saveFile);
    std::cout << "Saved!" << std::endl;
    std::cout << "SAVED num_correlations = " << num_correlations << std::endl;

    return;
}

// Find "closest" users and average their ratings of given movie
// Change this to: go through all users that have rated a certain movie
float kNN::predict(int user, int movie) {
    // TODO: get stats?
    std::priority_queue<corrUser> top_corr;
    // Put correlations between specified user and any other user
    // in a priority queue
    for (int user_i = 0; user_i < N_USERS; user_i++) {
        if (user_i > user) {
            break; // user_j > user_i > user
        }
        int start = corrMatrix[0][user_i];
        int end = corrMatrix[0][user_i + 1];
        for (int j = start; j < end; j++) {
            int user_j = corrMatrix[1][j];
            float c = corrMatrix[2][j];
            if (!isnan(c)) {
                int other_user = -1;
                if (user_i == user) {
                    other_user = user_j;
                }
                else if (user_j == user) {
                    other_user = user_i;
                }
                if (other_user > 0) {
                    // check if other_user has rated movie
                    int start_index = rowIndex[other_user];
                    int end_index = rowIndex[other_user + 1];
                    for (int i = start_index; i < end_index; ++i) {
                        if (columns[i] == movie) {
                            top_corr.push(corrUser(c, other_user, values[i]));
                        }
                        else if (columns[i] > movie) {
                            break;
                        }
                    }
                }
            }
        }
    }

    // get top K's average
    int K = 10;
    float total = 0;
    float weighted_total = 0;
    float sum_weights = 0;
    int actualK = 0;
    while (actualK < K) {
        if (top_corr.empty()) {
            break;
        }
        corrUser top = top_corr.top();
        top_corr.pop();

        // ignore negative correlations - remove after implementing weighted avg
        if (top.corr < 0) {
            break;
        }

        // std::cout << actualK << " top corr value = " << top.corr << " with user " << top.user << std::endl;

        total += float(top.rating);
        ++actualK;
    }

    if (actualK <= 0) {
        return baseline;
    }

    // for debugging purposes only
    // if (total != 0) {
    //     std::cout << "total = " << total << "\n";
    //     std::cout << "actualK = " << actualK << "\n";
    //     std::cout << "got a value!! rating = " << total / actualK << std::endl;
    // }
    if (user % 10 == 0) {
        std::cout << "predicting for user " << user << "\n";
    }

    return total / actualK;
}

void kNN::loadSaved(std::string fname) {
    FILE *in = fopen(fname.c_str(), "r");

    // Read num correlations
    int buf[3];
    fread(buf, sizeof(int), 3, in);
    num_correlations = buf[1];

    if (num_correlations == 0) {
        // TODO: handle special case
    }

    // Initialize correlation matrix
    for (int i = 0; i < 3; i++) {
        std::cout << "buf[" << i << "]" << " = " << buf[i] << "\n";
        std::vector<float> row(buf[i], 0);
        corrMatrix.push_back(row);
    }

    // Read matrix
    fread(&corrMatrix[0][0], sizeof(float), buf[0], in);
    fread(&corrMatrix[1][0], sizeof(float), buf[1], in);
    fread(&corrMatrix[2][0], sizeof(float), buf[2], in);
    fclose(in);
}

void kNN::save(std::string fname) {
    FILE *out = fopen(fname.c_str(), "wb");
    int buf[3];
    buf[0] = corrMatrix[0].size();
    buf[1] = corrMatrix[1].size();
    buf[2] = corrMatrix[2].size();
    fwrite(buf, sizeof(int), 3, out);

    fwrite(&corrMatrix[0][0], sizeof(float), corrMatrix[0].size(), out);
    fwrite(&corrMatrix[1][0], sizeof(float), corrMatrix[1].size(), out);
    fwrite(&corrMatrix[2][0], sizeof(float), corrMatrix[2].size(), out);
    fclose(out);
}

// rmse
float kNN::validate(std::string valid_file) {
    // load validation set
    load(valid_file);

    float sum_sq_error = 0;
    // predict and calculate
    for (int i = 0; i < numRatings; i++) {
        int user = ratings[i * DATA_POINT_SIZE + USER_IDX];
        int movie = ratings[i * DATA_POINT_SIZE + MOVIE_IDX];
        int actual_rating = ratings[i * DATA_POINT_SIZE + RATING_IDX];
        float rating = predict(user, movie);
        if (rating < 1) {
            rating = 1;
        }
        else if (rating > 5) {
            rating = 5;
        }

        sum_sq_error += (rating - actual_rating) * (rating - actual_rating);
    }

    return sqrt(sum_sq_error /= numRatings);
}

float kNN::rmse(float actual, float predicted) {
    return 0;
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
        + "_train=" + data_file;
    return fname;
}

int main(int argc, char **argv) {
    clock_t time0 = clock();
    kNN* knn = new kNN();
    // Baseline* base = new Baseline();

    std::string data_file = "4.dta";

    // Load data from file.
    knn->load(data_file);
    knn->baseline = 3; // TODO: figure out what to use

    // Normalize ratings
    // base->load(data_file);
    // base->train("unused variable");
    // knn->normalizeRatings(base->average_array, base->stdev_array);

    // Train by building correlation matrix
    knn->metric = kPearson;
    knn->shared_threshold = 2;
    knn->individual_threshold = 6;
    knn->K = 10;
    knn->train("model/knn/" + knn->getFilename(data_file) + ".save");

    clock_t time1 = clock();

    double kNN_ms = diffclock(time1, time0);
    std::cout << "kNN training took " << kNN_ms << " ms" << std::endl;

    // Validate
    std::cout << "Validating...\n";
    std::cout << "RMSE = " << knn->validate("2.dta") << "\n";
    std::cout << "Validation DONE\n";

    clock_t time_valid = clock();

    double valid_ms = diffclock(time_valid, time1);
    std::cout << "kNN validation took " << valid_ms << " ms" << std::endl;

    // Predict ratings and write to file
    // Load qual data
    knn->load(data_file); // can this be done more elegantly
    std::cout << "PREDICTIONS:\n";
    kNN* qual = new kNN();
    qual->load("5-1.dta"); // is this how we're going to do things

    // Prepare file
    std::ofstream outputFile;
    outputFile << std::setprecision(3);
    outputFile.open("out/" + knn->getFilename(data_file) + ".dta");

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
    std::cout << "kNN training took " << kNN_ms << " ms" << std::endl;
    std::cout << "kNN validation took " << valid_ms << " ms" << std::endl;
    double predict_ms = diffclock(time2, time_valid);
    std::cout << "kNN prediction took " << predict_ms << " ms" << std::endl;

    return 0;
}