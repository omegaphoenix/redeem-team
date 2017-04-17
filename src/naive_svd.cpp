#include "naive_svd.hpp"
#include <iostream>
#include <time.h>

// Clean up U, V.
NaiveSVD::~NaiveSVD() {
}

// Generic SGD training algorithm.
void NaiveSVD::train(void) {
}

// Computes one update step in SGD.
void NaiveSVD::update(void) {
}

int main(int argc, char **argv) {
    // Speed up stdio operations
    std::ios_base::sync_with_stdio(false);

    clock_t time0 = clock();
    NaiveSVD* nsvd = new NaiveSVD();
    clock_t time1 = clock();

    // Load in COO format into ratings vector
    nsvd->load();
    clock_t time2 = clock();
    double ms1 = diffclock(time1, time0);
    std::cout << "Initializing took " << ms1 << " ms" << std::endl;
    double ms2 = diffclock(time2, time1);
    std::cout << "Total loading took " << ms2 << " ms" << std::endl;
    double total_ms = diffclock(time2, time0);
    std::cout << "Total running time was " << total_ms << " ms" << std::endl;
    return 0;
}
