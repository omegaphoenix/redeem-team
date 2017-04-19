#include <iostream>
#include <time.h>
#include <math.h>
#include "baseline.hpp"

Baseline::~Baseline() {
    // None
}

float bogusMean() {
    
    return ;
}

float betterMean() {
    return ;
}


float globalAverage() {
    float sum = 0;
    float total = N_TRAINING;

    for (int i = 0; i < total; i++) {
        sum = sum + values[i]
    
    return sum/total;
}

void Baseline::train(std::string saveFile) {
    std::cout << "entered train()" << std::endl;
    build();
}

void loadSaved(std::string fname) {
    loadCSR(fname);
}

int main(int argc, char **argv) {
    clock_t time0 = clock();
    Baseline* baseline = new Baseline();
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
    baseline->predict(0, 0);

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