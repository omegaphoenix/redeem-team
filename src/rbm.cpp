#include "rbm.hpp"
#include <iostream>
#include <math.h>
#include "utils.hpp"
using namespace std;
using namespace utils;

// Initialize RBM variables.
RBM::RBM(int size, int numVis, int numHid, double **w, double *hb, double *vb) {
    N = size;
    nVisible = numVis;
    nHidden = numHid;

    if(w == NULL) {
        W = new double*[nHidden];
        for(int i=0; i<nHidden; i++) W[i] = new double[nVisible];
        double a = 1.0 / nVisible;

        for(int i=0; i<nHidden; i++) {
            for(int j=0; j<nVisible; j++) {
                W[i][j] = uniform(-a, a);
            }
        }
    } else {
        W = w;
    }

    if(hb == NULL) {
        hbias = new double[nHidden];
        for(int i=0; i<nHidden; i++) hbias[i] = 0;
    } else {
        hbias = hb;
    }

    if(vb == NULL) {
        vbias = new double[nVisible];
        for(int i=0; i<nVisible; i++) vbias[i] = 0;
    } else {
        vbias = vb;
    }
}

RBM::~RBM() {
    for(int i=0; i<nHidden; i++) delete[] W[i];
    delete[] W;
    delete[] hbias;
    delete[] vbias;
}

// Train RBM.
void RBM::constrastiveDivergence(int *input, double lr, int k) {
    double *phMean = new double[nHidden];
    int *phSample = new int[nHidden];
    double *nvMeans = new double[nVisible];
    int *nvSamples = new int[nVisible];
    double *nhMeans = new double[nHidden];
    int *nhSamples = new int[nHidden];

    // CD-k
    sampleHGivenV(input, phMean, phSample);

    for(int step=0; step<k; step++) {
        if(step == 0) {
            gibbsHvh(phSample, nvMeans, nvSamples, nhMeans, nhSamples);
        } else {
            gibbsHvh(nhSamples, nvMeans, nvSamples, nhMeans, nhSamples);
        }
    }

    for(int i=0; i<nHidden; i++) {
        for(int j=0; j<nVisible; j++) {
            // W[i][j] += lr * (phSample[i] * input[j] - nhMeans[i] * nvSamples[j]) / N;
            W[i][j] += lr * (phMean[i] * input[j] - nhMeans[i] * nvSamples[j]) / N;
        }
        hbias[i] += lr * (phSample[i] - nhMeans[i]) / N;
    }

    for(int i=0; i<nVisible; i++) {
        vbias[i] += lr * (input[i] - nvSamples[i]) / N;
    }

    delete[] phMean;
    delete[] phSample;
    delete[] nvMeans;
    delete[] nvSamples;
    delete[] nhMeans;
    delete[] nhSamples;
}

void RBM::sampleHGivenV(int *v0Sample, double *mean, int *sample) {
    for(int i=0; i<nHidden; i++) {
        mean[i] = propUp(v0Sample, W[i], hbias[i]);
        sample[i] = binomial(1, mean[i]);
    }
}

void RBM::sampleVGivenH(int *h0Sample, double *mean, int *sample) {
    for(int i=0; i<nVisible; i++) {
        mean[i] = propDown(h0Sample, i, vbias[i]);
        sample[i] = binomial(1, mean[i]);
    }
}

double RBM::propUp(int *v, double *w, double b) {
    double preSigmoidActivation = 0.0;
    for(int j=0; j<nVisible; j++) {
        preSigmoidActivation += w[j] * v[j];
    }
    preSigmoidActivation += b;
    return sigmoid(preSigmoidActivation);
}

double RBM::propDown(int *h, int i, double b) {
    double preSigmoidActivation = 0.0;
    for(int j=0; j<nHidden; j++) {
        preSigmoidActivation += W[j][i] * h[j];
    }
    preSigmoidActivation += b;
    return sigmoid(preSigmoidActivation);
}

void RBM::gibbsHvh(int *h0Sample, double *nvMeans, int *nvSamples, \
        double *nhMeans, int *nhSamples) {
    sampleVGivenH(h0Sample, nvMeans, nvSamples);
    sampleHGivenV(nvSamples, nhMeans, nhSamples);
}

void RBM::reconstruct(int *v, double *reconstructedV) {
    double *h = new double[nHidden];
    double preSigmoidActivation;

    for(int i=0; i<nHidden; i++) {
        h[i] = propUp(v, W[i], hbias[i]);
    }

    for(int i=0; i<nVisible; i++) {
        preSigmoidActivation = 0.0;
        for(int j=0; j<nHidden; j++) {
            preSigmoidActivation += W[j][i] * h[j];
        }
        preSigmoidActivation += vbias[i];

        reconstructedV[i] = sigmoid(preSigmoidActivation);
    }

    delete[] h;
}


void testRBM() {
    srand(0);

    double learningRate = 0.1;
    int trainingEpochs = 1000;
    int k = 1;

    int trainN = 6;
    int testN = 2;
    int nVisible = 6;
    int nHidden = 3;

    // training data
    int trainX[6][6] = {
        {1, 1, 1, 0, 0, 0},
        {1, 0, 1, 0, 0, 0},
        {1, 1, 1, 0, 0, 0},
        {0, 0, 1, 1, 1, 0},
        {0, 0, 1, 0, 1, 0},
        {0, 0, 1, 1, 1, 0}
    };


    // construct RBM
    RBM rbm(trainN, nVisible, nHidden, NULL, NULL, NULL);

    // train
    for(int epoch=0; epoch<trainingEpochs; epoch++) {
        for(int i=0; i<trainN; i++) {
            rbm.constrastiveDivergence(trainX[i], learningRate, k);
        }
    }

    // test data
    int testX[2][6] = {
        {1, 1, 0, 0, 0, 0},
        {0, 0, 0, 1, 1, 0}
    };
    double reconstructedX[2][6];


    // test
    for(int i=0; i<testN; i++) {
        rbm.reconstruct(testX[i], reconstructedX[i]);
        for(int j=0; j<nVisible; j++) {
            printf("%.5f ", reconstructedX[i][j]);
        }
        cout << endl;
    }

}



int main() {
    srand(0);
    return 0;
}
