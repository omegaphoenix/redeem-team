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
    debugPrint("Initializing RBM...\n");
    unsigned int i;

    // Number of full steps to run Gibb's sampler
    T = 1;

    // Initial learning rates
    epsilonW = 0.02;
    epsilonVB = 0.02;
    epsilonHB = 0.02;

    // Initialize W
    W = new float[N_MOVIES * N_FACTORS * MAX_RATING];
    for (i = 0; i < N_MOVIES * N_FACTORS * MAX_RATING; ++i) {
        W[i] = normalRandom() * 0.1;
    }

    // Initialize hidden units
    hidVars = new std::bitset<N_USERS * N_FACTORS>(0);
    hidProbs = new float[N_USERS * N_FACTORS];
    for (i = 0; i < N_USERS * N_FACTORS; ++i) {
        setHidVar(i, rand() % 2);
    }

    // Initialize V
    indicatorV = new std::bitset<N_MOVIES * MAX_RATING>[N_USERS];
    visProbs = new float[(long) N_USERS * N_MOVIES * MAX_RATING];
    for (i = 0; i < N_USERS; ++i) {
        indicatorV[i] = std::bitset<N_MOVIES * MAX_RATING>(0);
    }

    // Initialize feature biases
    hidBiases = new float[N_FACTORS];
    for (i = 0; i < N_FACTORS; ++i) {
        hidBiases[i] = uniform(0, 1);
    }
    visBiases = new float[N_MOVIES * MAX_RATING];

    // Initialize deltas
    dW = new float[N_MOVIES * N_FACTORS * MAX_RATING];
    dHidBiases = new float[N_FACTORS];
    dVisBiases = new float[N_MOVIES * MAX_RATING];

    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("RBM initialization took %f ms\n", ms1);
}

// Free memory
RBM::~RBM() {
    delete[] W;
    delete[] dW;
    delete[] hidBiases;
    delete[] dHidBiases;
    delete[] visBiases;
    delete[] dVisBiases;
    delete[] hidProbs;
    delete[] visProbs;
    delete hidVars;
    delete[] indicatorV;
}

// Load data
void RBM::init() {
    load("1.dta");
    clock_t time0 = clock();
    debugPrint("Initializing visible biases...\n");

    // Init visible biases to logs of respective base rates over all users
    unsigned int movie, rating, i;
    for (i = 0; i < numRatings; ++i) {
        movie = columns[i];
        rating = values[i];
        (visBiases[movie * MAX_RATING + rating])++;
    }
    clock_t time1 = clock();
    for (i = 0; i < N_MOVIES * MAX_RATING; ++i) {
        visBiases[i] = log(visBiases[i] / N_USERS);
    }
    clock_t time2 = clock();

    float ms1 = diffclock(time1, time0);
    float ms2 = diffclock(time2, time1);
    printf("Adding visible biases took %f ms\n", ms1);
    printf("Logging biases took %f ms\n", ms2);
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
    debugPrint("Resetting deltas...\n");
    clock_t time0 = clock();
    unsigned int i;
    for (i = 0; i < N_MOVIES * N_FACTORS * MAX_RATING; ++i) {
        dW[i] = 0;
    }
    for (i = 0; i < N_FACTORS; ++i) {
        dHidBiases[i] = 0;
    }
    for (i = 0; i < N_MOVIES * MAX_RATING; ++i) {
        dVisBiases[i] = 0;
    }
    clock_t time1 = clock();

    float ms1 = diffclock(time1, time0);
    printf("Resetting deltas took %f ms\n", ms1);
}

// Calculate the gradient averaged over all users
// TODO: Add biases
void RBM::calcGrad() {
    debugPrint("Calculating gradient...\n");
    clock_t time0 = clock();
    resetDeltas(); // set deltas back to zeros
    posStep(); // Calculate <vikhj>_data
    // TODO: Do we need to reset V?
    negStep(); // Calculate <vikhj>_T
    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Calculating gradient took %f ms\n", ms1);
}

void RBM::posStep() {
    debugPrint("Positive step...\n");
    clock_t time0 = clock();
    float dataVal;
    unsigned int userStartIdx, userEndIdx, movIdx, facIdx, idx, i, j, k, n, colIdx;
    int movFac = N_FACTORS * MAX_RATING;

    calcHidProbsUsingData();
    updateH();
    // First half of equation 6 in RBM for CF, Salakhutdinov 2007
    for (n = 0; n < N_USERS; ++n) {
        userStartIdx = rowIndex[n];
        userEndIdx = rowIndex[n + 1];
        for (colIdx = userStartIdx; colIdx < userEndIdx;
                colIdx++) {
            i = (int) columns[colIdx];
            movIdx = i * movFac;
            for (j = 0; j < N_FACTORS; ++j) {
                facIdx = j * MAX_RATING;
                for (k = 0; k < MAX_RATING; ++k) {
                    idx = movIdx + facIdx + k;
                    dataVal = getActualVal(n, i, j, k);
                    dW[idx] += dataVal;
                }
            }
        }
    }
    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Positive step took %f ms\n", ms1);
}

void RBM::negStep() {
    debugPrint("Negative step...\n");
    clock_t time0 = clock();
    float expectVal;
    unsigned int userStartIdx, userEndIdx, movIdx, facIdx, idx, i, j, k, n, colIdx;
    int movFac = N_FACTORS * MAX_RATING;

    // Update H and V several times
    for (unsigned int t = 0; t < T; t++) {
        runGibbsSampler();
    }

    // Second half of equation 6 in RBM for CF, Salakhutdinov 2007
    for (n = 0; n < N_USERS; ++n) {
        userStartIdx = rowIndex[n];
        userEndIdx = rowIndex[n + 1];
        for (colIdx = userStartIdx; colIdx < userEndIdx;
                colIdx++) {
            i = (int) columns[colIdx];
            movIdx = i * movFac;
            for (j = 0; j < N_FACTORS; ++j) {
                facIdx = j * MAX_RATING;
                for (k = 0; k < MAX_RATING; ++k) {
                    idx = movIdx + facIdx + k;
                    expectVal = getExpectVal(n, i, j, k);
                    dW[idx] -= expectVal;
                }
            }
        }
    }
    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Negative step took %f ms\n", ms1);
}

// Update W using contrastive divergence
void RBM::updateW() {
    debugPrint("Updating W...\n");
    clock_t time0 = clock();
    calcGrad();
    // Update W
    for (unsigned int i = 0; i < N_MOVIES * N_FACTORS * MAX_RATING; ++i) {
        W[i] += epsilonW * dW[i] / N_USERS;
    }
    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Updating W took %f ms\n", ms1);
}

// Update binary hidden states
void RBM::updateH() {
    debugPrint("Updating H...\n");
    clock_t time0 = clock();
    bool var;
    unsigned int i;
    for (i = 0; i < N_USERS * N_FACTORS; ++i) {
        var = uniform(0, 1) > hidProbs[i];
        setHidVar(i, var);
    }
    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Updating H took %f ms\n", ms1);
}

// Update binary visible states
void RBM::updateV() {
    debugPrint("Updating V...\n");
    clock_t time0 = clock();
    // TODO: Do we update all the V's or just the learned ones?
    bool var;
    unsigned int idx, n, i, k, userStartIdx, userEndIdx, colIdx;
    for (n = 0; n < N_USERS; ++n) {
        userStartIdx = rowIndex[n];
        userEndIdx = rowIndex[n + 1];
        for (colIdx = userStartIdx; colIdx < userEndIdx;
                colIdx++) {
            i = (int) columns[colIdx]; // movie
            for (k = 0; k < MAX_RATING; ++k) {
                idx = n * N_MOVIES * MAX_RATING + i * MAX_RATING + k;
                var = uniform(0, 1) > hidProbs[idx];
                setV(n, i, k, var);
            }
        }
    }
    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Updating V took %f ms\n", ms1);
}

// Iteratively update V and h
void RBM::runGibbsSampler() {
    debugPrint("Running Gibbs sampler...\n");
    clock_t time0 = clock();
    calcVisProbs();
    updateV();
    calcHidProbs();
    updateH();
    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Running Gibbs sampler took %f ms\n", ms1);
}

// Use training data to calculate hidden probabilities
void RBM::calcHidProbsUsingData() {
    debugPrint("Calculating hidden probabilities using data...\n");
    clock_t time0 = clock();
    // Reset hidProbs to b_j
    unsigned int userStartIdx, userEndIdx, n, i, k, j, idx, colIdx;
    resetHidProbs();

    for (n = 0; n < N_USERS; ++n) {
        userStartIdx = rowIndex[n];
        userEndIdx = rowIndex[n + 1];
        for (colIdx = userStartIdx; colIdx < userEndIdx;
                colIdx++) {
            i = (int) columns[colIdx]; // movie
            k = values[colIdx]; // rating
            for (j = 0; j < N_FACTORS; ++j) {
                idx = i * N_FACTORS * MAX_RATING + j * MAX_RATING + k;
                hidProbs[n * N_FACTORS + j] += W[idx];
            }
        }
    }

    // Compute hidProbs by taking sigmoid
    compHidProbs();
    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Calculating hidden probs w/ data took %f ms\n", ms1);
}

// Use visible states to calculate hidden probabilities
void RBM::calcHidProbs() {
    debugPrint("Calculating hidden probabilities using visible states...\n");
    clock_t time0 = clock();
    // Reset hidProbs to b_j
    unsigned int n, i, k, j, idx, userStartIdx, userEndIdx, colIdx;
    resetHidProbs();

    for (n = 0; n < N_USERS; ++n) {
        userStartIdx = rowIndex[n];
        userEndIdx = rowIndex[n + 1];
        for (colIdx = userStartIdx; colIdx < userEndIdx;
                colIdx++) {
            i = (int) columns[colIdx]; // movie
            k = values[colIdx]; // rating
            // Add v_i^k W_ij^k
            if (getV(n, i, k)) {
                for (j = 0; j < N_FACTORS; ++j) {
                    idx = i * N_FACTORS * MAX_RATING + j * MAX_RATING + k;
                    hidProbs[n * N_FACTORS + j] += W[idx];
                }
            }
        }
    }

    // Compute hidProbs by taking sigmoid
    compHidProbs();
    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Calculating hidden probs w/ vis took %f ms\n", ms1);
}

// Reset hidProbs to b_j
void RBM::resetHidProbs() {
    debugPrint("Resetting hidden probabilities to biases...\n");
    clock_t time0 = clock();
    unsigned int userStartIdx, i;
    float* hidBiasesEnd = hidBiases + N_FACTORS;
    for (i = 0; i < N_USERS; ++i) {
        userStartIdx = i * N_FACTORS;
        std::copy(hidBiases, hidBiasesEnd, hidProbs + userStartIdx);
    }
    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Resetting hid probs took %f ms\n", ms1);
}

// Take sigmoid function to get probability
void RBM::compHidProbs() {
    debugPrint("Sigmoiding hidden probabilities...\n");
    clock_t time0 = clock();
    unsigned int i;
    for (i = 0; i < N_USERS * N_FACTORS; ++i) {
        hidProbs[i] = sigmoid(hidProbs[i]);
        assert(hidProbs[i] >= 0 && hidProbs[i] <= 1);
    }
    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Sigmoiding hid probs took %f ms\n", ms1);
}

// Use hidden states to calculate visible probabilities
void RBM::calcVisProbs() {
    debugPrint("Calculating visible probabilities...\n");
    clock_t time0 = clock();
    // Reset visProbs to b_ik
    resetVisProbs();

    // Get sum of W's
    sumVisProbs();

    // Compute visProbs
    sumToVisProbs();
    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Calculating vis probs took %f ms\n", ms1);
}

// Reset visProbs to b_ik
void RBM::resetVisProbs() {
    debugPrint("Resetting visible probabilities to biases...\n");
    clock_t time0 = clock();
    unsigned int n, idx;
    unsigned int movieRatings = N_MOVIES * MAX_RATING;
    float *visBiasesEnd = visBiases + movieRatings;
    for (n = 0; n < N_USERS; ++n) {
        idx = n * movieRatings;
        std::copy(visBiases, visBiasesEnd, visProbs + idx);
    }
    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Resetting vis probs took %f ms\n", ms1);
}

// Do summation for visible probability calculation
void RBM::sumVisProbs() {
    debugPrint("Summing weights for visible probabilities...\n");
    clock_t time0 = clock();
    unsigned int userStartIdx, userEndIdx, n, i, j, k, idx, vIdx, colIdx;
    for (n = 0; n < N_USERS; ++n) {
        userStartIdx = rowIndex[n];
        userEndIdx = rowIndex[n + 1];
        for (colIdx = userStartIdx; colIdx < userEndIdx;
                colIdx++) {
            i = (int) columns[colIdx]; // movie
            k = values[colIdx]; // rating
            for (k = 0 ; k < MAX_RATING; ++k) {
                for (j = 0; j < N_FACTORS; ++j) {
                    if (getHidVar(j)) {
                        idx = i * N_FACTORS * MAX_RATING + j * MAX_RATING + k;
                        vIdx = n * N_MOVIES * MAX_RATING + i * MAX_RATING + k;
                        visProbs[vIdx] += W[idx];
                    }
                }
            }
        }
    }
    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Summing weights for vis probs took %f ms\n", ms1);
}

// Take exponent and then calculate probability using ratios
void RBM::sumToVisProbs() {
    debugPrint("Exponentiating visible probabilities...\n");
    clock_t time0 = clock();
    unsigned int userStartIdx, userEndIdx, n, i, k, vIdx, colIdx;
    float denom = 0;
    for (n = 0; n < N_USERS; ++n) {
        userStartIdx = rowIndex[n];
        userEndIdx = rowIndex[n + 1];
        for (colIdx = userStartIdx; colIdx < userEndIdx;
                colIdx++) {
            i = (int) columns[colIdx]; // movie
            denom = 0;
            for (k = 0 ; k < MAX_RATING; ++k) {
                vIdx = n * N_MOVIES * MAX_RATING + i * MAX_RATING + k;
                visProbs[vIdx] = exp(visProbs[vIdx]);
                denom += visProbs[vIdx];
            }
            for (k = 0 ; k < MAX_RATING; ++k) {
                vIdx = n * N_MOVIES * MAX_RATING + i * MAX_RATING + k;
                visProbs[vIdx] = visProbs[vIdx] / denom;
                assert (visProbs[vIdx] >= 0 && visProbs[vIdx] <= 1);
            }
        }
    }
    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Exponentiating vis probs took %f ms\n", ms1);
}

// Frequency with which movie i with rating k and feature j are on together
// when the features are being driven by the observed user-rating data from
// the training set
float RBM::getActualVal(int n, int i, int j, int k) {
    float prod = 0;
    if (getV(n, i, k)) {
        int idx = n * N_FACTORS + j;
        prod = hidProbs[idx];
    }
    return prod;
}

// <v_i^k h_j>_T in equation 6
// Expectation with respect to the distribution defined by the model
float RBM::getExpectVal(int n, int i, int j, int k) {
    float prod = 0;
    if (getV(n, i, k)) {
        int idx = n * N_FACTORS + j;
        prod = hidProbs[idx];
    }
    return prod;
}

void RBM::train(std::string saveFile) {
    debugPrint("Training...\n");
    clock_t timeStart = clock();
    for (unsigned int epoch = 0; epoch < RBM_EPOCHS; epoch++) {
        printf("Starting epoch %d\n", epoch);
        clock_t time0 = clock();
        updateW();
        clock_t time1 = clock();
        float ms1 = diffclock(time1, time0);
        printf("Epoch %d took %f ms\n", epoch, ms1);
    }
    clock_t timeEnd = clock();
    float msTotal = diffclock(timeEnd, timeStart);
    printf("Training took %f ms\n", msTotal);
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
    float ms1 = diffclock(time1, time0);
    float ms2 = diffclock(time2, time1);
    float ms3 = diffclock(time3, time2);
    float total_ms = diffclock(time3, time0);
    printf("Initialization took %f ms\n", ms1);
    printf("Total loading took %f ms\n", ms2);
    printf("Training took %f ms\n", ms3);
    printf("Total running time was %f ms\n", total_ms);
    delete rbm;
    return 0;
}
