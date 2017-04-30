#include <iostream>
#include <time.h>
#include <math.h>
#include "baseline.hpp"

Baseline::Baseline() : Model() {
    this->K = 0;
    this->average_array = new float[N_USERS];
    this->ratings_count = new float[N_USERS];
    this->stdev_array = new float[N_USERS];
}

Baseline::~Baseline() {
    delete this->average_array;
    delete this->ratings_count;
    delete this->stdev_array;
}

void Baseline::setK(float constant) {
    this->K = constant;
}

void Baseline::betterMean() {
    if (K != 0) {
        global = globalAverage();
        std::cout << "Global Average = " << global << std::endl;
    }

    for (int i = 0; i < N_USERS; i++) {
        average_array[i] = 0;
        ratings_count[i] = 0;

        for (int j = rowIndex[i]; j < rowIndex[i+1]; j++) {
            average_array[i] = average_array[i] + float(values[j]);
            ratings_count[i]++;
        }
        average_array[i] = (K * global + average_array[i]) / (K + ratings_count[i]);
    }
}

void Baseline::standardDeviation() {
    for (int i = 0; i < N_USERS; i++) {
        stdev_array[i] = 0;

        for (int j = rowIndex[i]; j < rowIndex[i+1]; j++) {
            stdev_array[i] = pow(float(values[j]) - average_array[i], 2);
        }
        stdev_array[i] = stdev_array[i] / (ratings_count[i] - 1);
        stdev_array[i] = sqrt(stdev_array[i]);
    }      
}

float Baseline::globalAverage() {
    float sum = 0;

    for (int i = 0; i < N_TRAINING; i++) {
        sum = sum + values[i];
    }
    return sum/total;
}

void Baseline::train(std::string saveFile) {
    std::cout << "entered train()" << std::endl;
    betterMean();
    standardDeviation();
}

void Baseline::loadSaved(std::string fname) {
    //loadCSR(fname);
}

int main(int argc, char **argv) {
    // Check the number of parameters
    if (argc < 2) {
        // Tell the user how to run the program
        std::cerr << "Usage: " << argv[0] << "K" << std::endl;
        std::cerr << "Use K = 0 for regular mean" << std::endl;
        std::cerr << "For better mean, article recommends K = 25" << std::endl;  
        return 1;
    }

    clock_t time0 = clock();
    Baseline* baseline = new Baseline();
    baseline->setK(atof(argv[1]));
    std::cout << "K = " << baseline->K <<std::endl;
 
    clock_t time1 = clock();

    // Load data from file.
    baseline->load("1.dta");
    clock_t time2 = clock();

    // Train by building correlation matrix
    std::cout << "Begin training\n";
    baseline->train("unused variable");

    clock_t time3 = clock();

    // Predict ratings
    // Load qual data
    // baseline->predict(0, 0);

    // Write predictions to file

    // Output times.
    double ms1 = diffclock(time1, time0);
    std::cout << "Initialization took " << ms1 << " ms" << std::endl;
    double ms2 = diffclock(time2, time1);
    std::cout << "Loading took " << ms2 << " ms" << std::endl;
    double total_ms = diffclock(time2, time0);
    std::cout << "Total took " << total_ms << " ms" << std::endl;

    double baseline_ms = diffclock(time3, time0);
    std::cout << "baseline took " << baseline_ms << " ms" << std::endl;

    return 0;
}