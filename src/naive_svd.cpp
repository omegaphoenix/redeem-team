#include "naive_svd.hpp"
#include <iostream>
#include <time.h>

// Clean up U, V.
NaiveSVD::~NaiveSVD() {
}

void NaiveSVD::setParams(int K, float eta, float lambda) {
    this->K = K;
    this->eta = eta;
    this->lambda = lambda;
}

// Generic SGD training algorithm.
void NaiveSVD::train() {
}

// Run one epoch of SGD, returning delta error.
float NaiveSVD::runEpoch() {
}

// Computes one update step in SGD.
std::pair<float, float> NaiveSVD::update(int user, int movie, float rating) {
}

float NaiveSVD::computeError() {
}


void NaiveSVD::save(std::string fname) {
}

void NaiveSVD::loadSaved(std::string fname) {
}

void NaiveSVD::printOutput(std::string fname) {
}

// Returns the differences in ms.
static double diffclock(clock_t clock1, clock_t clock2) {
   double diffticks = clock1 - clock2;
   double diffms = (diffticks) / (CLOCKS_PER_SEC / 1000);
   return diffms;
}

int main(int argc, char **argv) {
    clock_t time0 = clock();
    NaiveSVD* nsvd = new NaiveSVD();
    clock_t time1 = clock();
    // Load data from file.
    nsvd->loadFresh("data/um/all.dta");
    clock_t time2 = clock();

    // Output times.
    double ms1 = diffclock(time1, time0);
    std::cout << "Initialization took " << ms1 << " ms" << std::endl;
    double ms2 = diffclock(time2, time1);
    std::cout << "Loading took " << ms2 << " ms" << std::endl;
    double total_ms = diffclock(time2, time0);
    std::cout << "Total took " << total_ms << " ms" << std::endl;
    return 0;
}
