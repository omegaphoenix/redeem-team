#include "naive_svd.hpp"

// Clean up U, V
NaiveSVD::~NaiveSVD() {
}

// Generic SGD training algorithm.
void NaiveSVD::train() {
}

// Computes one update step in SGD.
void NaiveSVD::update() {
}

int main(int argc, char **argv) {
    NaiveSVD* nsvd = new NaiveSVD;
    nsvd->loadFresh("data/um/all.dta");
    return 0;
}
