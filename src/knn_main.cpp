#include <fstream>
#include <iomanip>
#include <time.h>
#include "baseline.hpp"
#include "knn.hpp"

int main(int argc, char **argv) {
    // Process cmd line args
    if ((argc > 1 && argc < 4) || argc > 4) {
        std::cout << "Usage: knn [shared threshold] [individual threshold] [number of neighbors]\n";
        return 1;
    }
    else if (argc == 4) {
        std::cout << "Running kNN with i = " << argv[1] << ", s = " << argv[2]
        << ", k = " << argv[3] << "\n";
    }
    else {
        std::cout << "Running kNN with default params\n";
    }

    clock_t time0 = clock();
    kNN* knn = new kNN();
    Baseline* base = new Baseline();

    std::string data_file = "1.dta";

    // Load data from file.
    knn->load(data_file);
    knn->transposeMU();
    knn->metric = kPearson;

    // Set kNN parameters
    if (argc == 4) {
        knn->shared_threshold = atoi(argv[1]);
        knn->individual_threshold = atoi(argv[2]);
        knn->K = atoi(argv[3]);
    }
    else {
        //or change variables here for testing
        knn->shared_threshold = 6;
        knn->individual_threshold = 1800;
        knn->K = 100;
    }

    // Get baseline and maybe normalize ratings
    base->load(data_file);
    base->train("unused variable");
    knn->avg_array = base->average_array;
    knn->stdev_array = base->stdev_array;

    // Train by building correlation matrix
    knn->train("model/knn/" + knn->getFilename(data_file) + ".save");

    clock_t time1 = clock();

    double kNN_ms = diffclock(time1, time0);
    std::cout << "kNN training took " << kNN_ms << " ms" << std::endl;

    // Validate [begin]
    std::cout << "Validating...\n";
    kNN* valid = new kNN();
    valid->load("2.dta");
    std::cout << "RMSE = \n" << knn->validate(valid->ratings, valid->numRatings) << "\n";
    std::cout << "Validation DONE\n";
    // Validate [end]

    clock_t time_valid = clock();

    double valid_ms = diffclock(time_valid, time1);
    std::cout << "kNN validation took " << valid_ms << " ms" << std::endl;

    // Predict ratings and write to file [begin]
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

        outputFile << rating << "\n";
    }
    outputFile.close();
    // Predict [end]

    clock_t time2 = clock();

    // Output times.
    std::cout << "kNN training took " << kNN_ms << " ms" << std::endl;
    std::cout << "kNN validation took " << valid_ms << " ms" << std::endl;
    double predict_ms = diffclock(time2, time_valid);
    std::cout << "kNN prediction took " << predict_ms << " ms" << std::endl;

    return 0;
}