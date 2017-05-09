#include "rbm.hpp"
#include <assert.h>
#include <cmath>
#include <float.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
using namespace std;

// Initialize RBM variables.
RBM::RBM() {
    clock_t time0 = clock();
    printf("Initializing RBM...\n");

    // Number of full steps to run Gibb's sampler
    T = 1;

    // Initial learning rates
    epsilonW = 0.02;
    epsilonVB = 0.02;
    epsilonHB = 0.02;

    // Initialize W
    W = new double[N_MOVIES * N_FACTORS * MAX_RATING];
    for (unsigned int i = 0; i < N_MOVIES * N_FACTORS * MAX_RATING; ++i) {
        W[i] = normalRandom() * 0.1;
    }

    // Initialize hidden units
    hidVars = new std::bitset<N_USERS * N_FACTORS>(0);
    hidProbs = new double[N_USERS * N_FACTORS];
    for (unsigned int i = 0; i < N_USERS * N_FACTORS; ++i) {
        setHidVar(i, rand() % 2);
        hidProbs[i] = uniform(0, 1);
    }

    // Initialize V
    indicatorV = new std::bitset<N_MOVIES * MAX_RATING>[N_USERS];
    for (unsigned int i = 0; i < N_USERS; ++i) {
        indicatorV[i] = std::bitset<N_MOVIES * MAX_RATING>(0);
    }

    // Initialize feature biases
    hidBiases = new double[N_FACTORS];
    for (unsigned int i = 0; i < N_FACTORS; ++i) {
        hidBiases[i] = uniform(0, 1);
    }
    // Init visible biases to logs of respective base rates over all users
    visBiases = new double[N_MOVIES * MAX_RATING];
    visProbs = new double[N_MOVIES * MAX_RATING];
    unsigned int movie, rating;
    for (unsigned int i = 0; i < numRatings; ++i) {
        movie = columns[i];
        rating = values[i];
        (visBiases[movie * MAX_RATING + rating])++;
    }
    for (unsigned int i = 0; i < N_MOVIES * MAX_RATING; ++i) {
        visBiases[i] = log(visBiases[i] / N_USERS);
    }

    // Initialize deltas
    dW = new double[N_MOVIES * N_FACTORS * MAX_RATING];
    dHidBiases = new double[N_FACTORS];
    dVisBiases = new double[N_MOVIES * MAX_RATING];

    clock_t time1 = clock();
    double ms1 = diffclock(time1, time0);
    std::cout << "RBM initialization took " << ms1 << " ms" << std::endl;
}

// Free memory
RBM::~RBM() {
    delete[] W;
    delete[] dW;
    delete hidVars;
    delete[] hidProbs;
    delete[] hidBiases;
    delete[] visBiases;
    delete[] indicatorV;
}

// Load data
void RBM::init() {
    load("3.dta");
}

// Set nth hidden variable to newVal
void RBM::setHidVar(int nthHidVar, bool newVal) {
    hidVars->set(nthHidVar, newVal);
}

// Get nth hidden variable
bool RBM::getHidVar(int nthHidVar) {
    // Need to get the 0th element first since it is a pointer
    return (*hidVars)[nthHidVar];
}

// Set the nth user's kth rating for the ith movie
void RBM::setV(int n, int i, int k, bool newVal) {
    int idx = i * MAX_RATING + k;
    indicatorV[n].set(idx, newVal);
}

// Did the nth user rate the ith movie as k
bool RBM::getV(int n, int i, int k) {
    // Need to get the 0th element first since it is a pointer
    int idx = i * MAX_RATING + k;
    return indicatorV[n][idx];
}

// Set all deltas to 0
// TODO: Consider just creating a new array for speed
void RBM::resetDeltas() {
    for (unsigned int i = 0; i < N_MOVIES * N_FACTORS * MAX_RATING; ++i) {
        dW[i] = 0;
    }
    for (unsigned int i = 0; i < N_FACTORS; ++i) {
        dHidBiases[i] = 0;
    }
    for (unsigned int i = 0; i < N_MOVIES * MAX_RATING; ++i) {
        dVisBiases[i] = 0;
    }
}

// Calculate the gradient averaged over all users
// TODO: Add biases
// TODO: Split into positive and negative steps
void RBM::calcGrad() {
    resetDeltas(); // set deltas back to zeros
    posStep(); // Calculate <vikhj>_data
    // TODO: Do we need to reset V?
    negStep(); // Calculate <vikhj>_T
}

void RBM::posStep() {
    double dataVal;
    unsigned int userStartIdx, userEndIdx, movIdx, facIdx, idx, i;
    int movFac = N_FACTORS * MAX_RATING;

    calcHidProbsUsingData();
    updateH();
    // First half of equation 6 in RBM for CF, Salakhutdinov 2007
    for (unsigned int n = 0; n < N_USERS; ++n) {
        userStartIdx = rowIndex[n];
        userEndIdx = rowIndex[n + 1];
        for (unsigned int colIdx = userStartIdx; colIdx < userEndIdx;
                colIdx++) {
            i = (int) columns[colIdx];
            movIdx = i * movFac;
            for (unsigned int j = 0; j < N_FACTORS; ++j) {
                facIdx = j * MAX_RATING;
                for (unsigned int k = 0; k < MAX_RATING; ++k) {
                    idx = movIdx + facIdx + k;
                    dataVal = getActualVal(n, i, j, k);
                    dW[idx] += dataVal;
                }
            }
        }
    }
}

void RBM::negStep() {
    double expectVal;
    unsigned int userStartIdx, userEndIdx, movIdx, facIdx, idx, i;
    int movFac = N_FACTORS * MAX_RATING;
    // TODO Update H and V several times
    for (unsigned int t = 0; t < T; t++) {
        runGibbsSampler();
    }
    // Second half of equation 6 in RBM for CF, Salakhutdinov 2007
    for (unsigned int n = 0; n < N_USERS; ++n) {
        userStartIdx = rowIndex[n];
        userEndIdx = rowIndex[n + 1];
        for (unsigned int colIdx = userStartIdx; colIdx < userEndIdx;
                colIdx++) {
            i = (int) columns[colIdx];
            movIdx = i * movFac;
            for (unsigned int j = 0; j < N_FACTORS; ++j) {
                facIdx = j * MAX_RATING;
                for (unsigned int k = 0; k < MAX_RATING; ++k) {
                    idx = movIdx + facIdx + k;
                    expectVal = getExpectVal(n, i, j, k);
                    dW[idx] -= expectVal;
                }
            }
        }
    }
}

// Update W using contrastive divergence
void RBM::updateW() {
    calcGrad();
    // Update W
    for (unsigned int i = 0; i < N_MOVIES * N_FACTORS * MAX_RATING; ++i) {
        W[i] += epsilonW * dW[i] / N_USERS;
    }
}

// Update binary hidden states
void RBM::updateH() {
    bool var;
    for (int i = 0; i < N_USERS * N_FACTORS; ++i) {
        var = uniform(0, 1) > hidProbs[i];
        setHidVar(i, var);
    }
}

// Update binary visible states
void RBM::updateV() {
    // TODO: Do we update all the V's or just the learned ones?
    bool var;
    unsigned int idx;
    for (unsigned int n = 0; n < N_USERS; ++n) {
        for (unsigned int i = 0; i < N_MOVIES; ++i) {
            for (unsigned int k = 0; k < MAX_RATING; ++k) {
                idx = n * N_MOVIES * MAX_RATING + i * MAX_RATING + k;
                var = uniform(0, 1) > hidProbs[idx];
                setV(n, i, k, var);
            }
        }
    }
}

// Iteratively update V and h
void RBM::runGibbsSampler() {
    calcVisProbs();
    updateV();
    calcHidProbs();
    updateH();
}

// Use training data to calculate hidden probabilities
void RBM::calcHidProbsUsingData() {
    // Reset hidProbs to b_j
    unsigned int userStartIdx, userEndIdx, i, k, idx;
    for (i = 0; i < N_USERS; ++i) {
        userStartIdx = i * N_FACTORS;
        std::copy(hidBiases, hidBiases + N_FACTORS, hidProbs + userStartIdx);
    }

    for (unsigned int n = 0; n < N_USERS; ++n) {
        userStartIdx = rowIndex[n];
        userEndIdx = rowIndex[n + 1];
        for (unsigned int colIdx = userStartIdx; colIdx < userEndIdx;
                colIdx++) {
            i = (int) columns[colIdx]; // movie
            k = values[colIdx]; // rating
            for (unsigned int j = 0; j < N_FACTORS; ++j) {
                idx = i * N_FACTORS * MAX_RATING + j * MAX_RATING + k;
                hidProbs[n * N_FACTORS + j] += W[idx];
            }
        }
    }

    // Compute hidProbs
    for (i = 0; i < N_USERS * N_FACTORS; ++i) {
        hidProbs[i] = sigmoid(hidProbs[i]);
        assert(hidProbs[i] >= 0 && hidProbs[i] <= 1);
    }
}

// Use visible states to calculate hidden probabilities
void RBM::calcHidProbs() {
    // Reset hidProbs to b_j
    unsigned int userStartIdx, i, k, idx;
    for (i = 0; i < N_USERS; ++i) {
        userStartIdx = i * N_FACTORS;
        std::copy(hidBiases, hidBiases + N_FACTORS, hidProbs + userStartIdx);
    }

    for (unsigned int n = 0; n < N_USERS; ++n) {
        for (i = 0; i < N_MOVIES; ++i) {
            for (k = 0; k < MAX_RATING; ++k) {
                // Add v_i^k W_ij^k
                if (getV(n, i, k)) {
                    for (unsigned int j = 0; j < N_FACTORS; ++j) {
                        idx = i * N_FACTORS * MAX_RATING + j * MAX_RATING + k;
                        hidProbs[n * N_FACTORS + j] += W[idx];
                    }
                }
            }
        }
    }

    // Compute hidProbs
    for (i = 0; i < N_USERS * N_FACTORS; ++i) {
        hidProbs[i] = sigmoid(hidProbs[i]);
        assert(hidProbs[i] >= 0 && hidProbs[i] <= 1);
    }
}

// Use hidden states to calculate visible probabilities
void RBM::calcVisProbs() {
}

// Frequency with which movie i with rating k and feature j are on together
// when the features are being driven by the observed user-rating data from
// the training set
double RBM::getActualVal(int n, int i, int j, int k) {
    double prod = 0;
    if (getV(n, i, k)) {
        int idx = n * N_MOVIES * MAX_RATING + i * MAX_RATING + k;
        prod = hidProbs[idx];
    }
    return prod;
}

// <v_i^k h_j>_{T} in equation 6
// Expectation with respect to the distribution defined by the model
double RBM::getExpectVal(int n, int i, int j, int k) {
    double prod = 0;
    if (getV(n, i, k)) {
        int idx = n * N_MOVIES * MAX_RATING + i * MAX_RATING + k;
        prod = hidProbs[idx];
    }
    return prod;
}

int main() {
    // Speed up stdio operations
    std::ios_base::sync_with_stdio(false);
    srand(0);

    clock_t time0 = clock();
    // Initialize
    RBM *rbm = new RBM();
    clock_t time1 = clock();
    rbm->init();
    clock_t time2 = clock();

    // Learn parameters
    rbm->train("data/um/rbm.save");
    clock_t time3 = clock();
    double ms1 = diffclock(time1, time0);
    std::cout << "Initializing took " << ms1 << " ms" << std::endl;
    double ms2 = diffclock(time2, time1);
    std::cout << "Total loading took " << ms2 << " ms" << std::endl;
    double ms3 = diffclock(time3, time2);
    std::cout << "Training took " << ms3 << " ms" << std::endl;
    double total_ms = diffclock(time3, time0);
    std::cout << "Total running time was " << total_ms << " ms" << std::endl;
    delete rbm;
    return 0;
}
