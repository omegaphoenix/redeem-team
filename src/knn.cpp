#include "baseline.hpp"
#include "knn.hpp"
#include <algorithm>
#include <assert.h>
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

/* Stores correlation value and correlated user. Use in predict, when you have
 * a specified user and specified movie. */
struct corrUser {
    float corr;
    int user; // user that is correlated with specified user
    int rating; // rating that user gave specified movie

    corrUser(float c, int u, int r) : corr(c), user(u), rating(r) {}

    bool operator<(const struct corrUser &other) const
    {
        return corr < other.corr;
    }
};

struct pearsonIntermediate {
    // for all movies rated by both user i and user j
    float i; // sum of ratings by user i
    float j; // sum of ratings by user j
    float ij; // sum of product of rating by user i and user j for each movie
    float ii; // sum of square of ratings by user i
    float jj; // sum of square of ratings by user j
    unsigned int count; // number of movies rated by both users

    pearsonIntermediate() : i(0), j(0), ij(0), ii(0), jj(0), count(0) {}
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

float kNN::denormalize(float normalized, double stdev, double ave) {
    return (normalized * stdev) + ave;
}

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
    // //sanity check corrMatrix
    // std::cout << "corrMatrix - rowIndex\n";
    // for (int i = 0; i < 300; i++) {
    //     std::cout << corrMatrix[0][i] << " ";
    // }
    // std::cout << "\ncorrMatrix - other user\n";
    // for (int i = 0; i < 10; i++) {
    //     std::cout << corrMatrix[1][i] << " ";
    // }
    // std::cout << "\ncorrMatrix - correlations\n";
    // for (int i = 0; i < 10; i++) {
    //     std::cout << corrMatrix[2][i] << " ";
    // }
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

    // Initialize corrMatrix with three rows
    // 0: index at which user i's correlations begin (int) 0-indexed
    // 1: user j (int) 0-indexed
    // 2: correlation (float)
    for (int i = 0; i < 3; i++) {
        std::vector<float> row;
        corrMatrix.push_back(row);
    }

    num_correlations = 0; // also acts as index
    int num_users = 0;
    int nan_coeff = 0;

    for (int user_i = 0; user_i < N_USERS; user_i++) {
        if (user_i % 10000 == 0) {
            std::cout << "calculating correlations for user " << user_i << "\n";
        }
        if (rowIndex[user_i + 1] - rowIndex[user_i] < individual_threshold) {
            corrMatrix[0].push_back(num_correlations);
            continue;
        }
        pearsonIntermediate* arr = new pearsonIntermediate[N_USERS];

        // for each movie rated by i
        for (int m = rowIndex[user_i]; m < rowIndex[user_i + 1]; m++) {
            int movie = columns[m];
            int rating_i = values[m];

            // for each j who also rated the movie
            for (int j = murowIndex[movie]; j < murowIndex[movie + 1]; j++) {
                int user_j = mucolumns[j];

                if (user_j <= user_i) {
                    continue;
                }

                if (rowIndex[user_j + 1] - rowIndex[user_j] >= individual_threshold) {
                    int rating_j = muvalues[j];

                    arr[user_j].i += rating_i;
                    arr[user_j].j += rating_j;
                    arr[user_j].ij += rating_i * rating_j;
                    arr[user_j].ii += rating_i * rating_i;
                    arr[user_j].jj += rating_j * rating_j;
                    arr[user_j].count++;
                }
            }
        }

        // Go through arr and calculate correlations
        corrMatrix[0].push_back(num_correlations);
        for (int j = 0; j < N_USERS; j++) {
            if (arr[j].count >= shared_threshold) {
                float pearson = (arr[j].ij / arr[j].count
                    - (arr[j].i / arr[j].count) * (arr[j].j / arr[j].count))
                / sqrt(
                    ((arr[j].ii / arr[j].count)
                        - (arr[j].i / arr[j].count) * (arr[j].i / arr[j].count))
                    * ((arr[j].jj / arr[j].count)
                        - (arr[j].j / arr[j].count) * (arr[j].j / arr[j].count))
                    );
                if (pearson == 0) {
                    // std::cout << "pearson coeff is 0\n";
                    continue;
                }
                else if (isnan(pearson)) {
                    // std::cout << "pearson coeff is nan\n";
                    nan_coeff++;
                    continue;
                }
                else {
                    corrMatrix[1].push_back(j);
                    corrMatrix[2].push_back(pearson);
                    num_correlations++;
                    if (num_correlations % 10000 == 0) {
                        std::cout << "number of correlations: " << num_correlations << "\n";
                    }
                }
            }
        }
        delete[] arr;
    }
    corrMatrix[0].push_back(num_correlations);

    std::cout<< "Number of nan coeffs = " << nan_coeff << "\n";
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

    int movie_start = murowIndex[movie];
    int movie_end = murowIndex[movie + 1];

    int other_user;
    int first_user;
    int second_user;

    // Get each OTHER USER that has rated MOVIE
    for (int movie_idx = movie_start; movie_idx < movie_end; movie_idx++) {
        int other_user = mucolumns[movie_idx];

        // to look up correlation, we need first_user < second_user
        //assert(other_user != user);
        if (other_user == user) {
            // std::cout << "(317, 4498) --> " << ratings[];
            std::cout << "user " << user << " already rated movie " << movie
            << "! gave it a " << muvalues[movie_idx] << "\n";
            exit(0);
        }
        first_user = (user < other_user) ? user : other_user;
        second_user = (user > other_user) ? user : other_user;

        // std::cout << "first user = " << first_user << "\n";
        // std::cout << "second user = " << second_user << "\n";

        // Retrieve correlation from matrix
        // TODO: consider binary search?
        for (int i = corrMatrix[0][first_user];
            i < corrMatrix[0][first_user + 1];
            i++) {
            if (corrMatrix[1][i] == second_user) {
                // std::cout << "found nonzero correlation!\n";
                // Retrieve movie rating by OTHER USER
                int rating = getRatingCSR(other_user, movie);
                if (rating > 0) {
                    top_corr.push(corrUser(
                        corrMatrix[2][i], other_user, float(rating)));
                }
                break;
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

    std::cout << "Reading file...\n";
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
float kNN::validate(int* valid_ratings, int valid_numRatings) {
    float sum_sq_error = 0;
    // predict and calculate
    for (int i = 0; i < valid_numRatings; i++) {
        int user = valid_ratings[i * DATA_POINT_SIZE + USER_IDX];
        int movie = valid_ratings[i * DATA_POINT_SIZE + MOVIE_IDX];
        // std::cout << "valid // user = " << user << "\n";
        // std::cout << "valid // movie = " << movie << "\n";
        int actual_rating = valid_ratings[i * DATA_POINT_SIZE + RATING_IDX];
        float rating = predict(user, movie);
        if (rating < 1) {
            rating = 1;
        }
        else if (rating > 5) {
            rating = 5;
        }

        sum_sq_error += (rating - actual_rating) * (rating - actual_rating);
    }

    return sqrt(sum_sq_error /= valid_numRatings);
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

// TODO: consider binary search?
int kNN::getRatingCSR(int user, int movie) {
    int start_index = rowIndex[user - 1];
    int end_index = rowIndex[user];
    for (int i = start_index; i < end_index; ++i) {
        if (columns[i] == movie) {
            return values[i];
        }
        else if (columns[i] > movie) {
            return -1;
        }
    }
}

int main(int argc, char **argv) {
    clock_t time0 = clock();
    kNN* knn = new kNN();
    // Baseline* base = new Baseline();

    std::string data_file = "1.dta";

    // Load data from file.
    knn->load(data_file);
    knn->transposeMU();

    // Normalize ratings
    // base->load(data_file);
    // base->train("unused variable");
    // knn->normalizeRatings(base->average_array, base->stdev_array);

    // Train by building correlation matrix
    knn->metric = kPearson;
    knn->shared_threshold = 6;
    knn->individual_threshold = 1800;
    knn->K = 10;
    knn->baseline = 3; // TODO: figure out what to use
    knn->train("model/knn/" + knn->getFilename(data_file) + ".save");

    clock_t time1 = clock();

    double kNN_ms = diffclock(time1, time0);
    std::cout << "kNN training took " << kNN_ms << " ms" << std::endl;

    // Validate
    std::cout << "Validating...\n";
    kNN* valid = new kNN();
    valid->load("2.dta");
    std::cout << "RMSE = " << knn->validate(valid->ratings, valid->numRatings) << "\n";
    std::cout << "Validation DONE\n";

    clock_t time_valid = clock();

    double valid_ms = diffclock(time_valid, time1);
    std::cout << "kNN validation took " << valid_ms << " ms" << std::endl;

    // Predict ratings and write to file
    // Load qual data
    // knn->load(data_file); // can this be done more elegantly
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
        // std::cout << "qual // user = " << user << "\n";
        // std::cout << "qual // movie = " << movie << "\n";
        float rating = knn->predict(user, movie);
        if (rating < 1) {
            rating = 1;
        }
        else if (rating > 5) {
            rating = 5;
        }

        outputFile << rating << "\n";
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