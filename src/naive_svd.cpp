#include "naive_svd.hpp"
#include <iostream>
#include <time.h>

// Clean up U, V.
NaiveSVD::~NaiveSVD() {
}

// Generic SGD training algorithm.
void NaiveSVD::train() {
}

// Computes one update step in SGD.
void NaiveSVD::update() {
}

// Returns the differences in ms.
static double diffclock(clock_t clock1, clock_t clock2) {
  double diffticks = clock1 - clock2;
  double diffms = (diffticks) / (CLOCKS_PER_SEC / 1000);
  return diffms;
}

int main(int argc, char **argv) {
    clock_t time0 = clock();

    // Initialize ratings data structure.
    NaiveSVD* nsvd = new NaiveSVD();
    clock_t time1 = clock();

    // Load data from file.
    nsvd->loadFresh("data/um/1.dta");
    clock_t time2 = clock();

    // Output ratings in new format.
    std::cout << "Outputing ratings" << std::endl;
    nsvd->outputRatingsCSR("data/um/test_csr.dta");
    clock_t time3 = clock();

    // Output times.
    double ms1 = diffclock(time1, time0);
    std::cout << "Initialization took " << ms1 << " ms" << std::endl;
    double ms2 = diffclock(time2, time1);
    std::cout << "Loading took " << ms2 << " ms" << std::endl;
    double ms3 = diffclock(time3, time2);
    std::cout << "Converting data took " << ms3 << " ms" << std::endl;
    double total_ms = diffclock(time3, time0);
    std::cout << "Total took " << total_ms << " ms" << std::endl;
    return 0;
}
