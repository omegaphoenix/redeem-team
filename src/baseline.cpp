#include <iostream>
#include <time.h>
#include <math.h>
#include "baseline.hpp"

Baseline::Baseline() : Model() {
    this->K = 0;
    this->average_array = new double[N_USERS];
    this->ratings_count = new float[N_USERS];
    this->stdev_array = new double[N_USERS];
    this->movie_average_array = new double[N_MOVIES];
    this->movie_count = new float[N_MOVIES];

    for (int i = 0; i < N_MOVIES;i++) {
        movie_count[i] = 0;
        movie_average_array[i] = 0;
    }
}

Baseline::~Baseline() {
    delete this->average_array;
    delete this->ratings_count;
    delete this->stdev_array;
    delete this->movie_average_array;
}

void Baseline::setK(float constant) {
    this->K = constant;
}

void Baseline::betterMean() {
    for (int i = 0; i < N_USERS; i++) {
        average_array[i] = 0;
        ratings_count[i] = 0;

        for (int j = rowIndex[i]; j < rowIndex[i+1]; j++) {
            average_array[i] = average_array[i] + float(values[j]);
            ratings_count[i]++;
        }
        if (ratings_count[i] != 0) {
            average_array[i] = (K * global + average_array[i]) / (K + ratings_count[i]);
        }

        if (isnan(average_array[i])) {
            std::cout << "user ID: " << i << std::endl;
        }
    }
}

void Baseline::movieMean() {
    for (int i = 0; i < numRatings; i++) {
        int index = columns[i];
        movie_average_array[index] = movie_average_array[index] + float(values[i]);
        movie_count[index]++;
    }

    for (int i = 0; i < N_MOVIES;i++) {
        if (movie_count[i] != 0) {
            movie_average_array[i] = movie_average_array[i]/ movie_count[i];
        }
    }
}


void Baseline::standardDeviation() {
    for (int i = 0; i < N_USERS; i++) {
        stdev_array[i] = 0;

        for (int j = rowIndex[i]; j < rowIndex[i+1]; j++) {
            stdev_array[i] = pow(float(values[j]) - average_array[i], 2);
        }
        stdev_array[i] = stdev_array[i] / ratings_count[i];
        stdev_array[i] = sqrt(stdev_array[i]);
    }
}

void Baseline::globalAverage() {
    double sum = 0;

    for (int i = 0; i < numRatings; i++) {
        sum = sum + values[i];
    }

    global = sum / numRatings;
    std::cout << "Global Average: " << global << std::endl;
}

void Baseline::train(std::string saveFile) {
    std::cout << "entered train()" << std::endl;
    globalAverage();
    betterMean();
    standardDeviation();
    movieMean();
}

void Baseline::loadSaved(std::string fname) {
    //loadCSR(fname);
}