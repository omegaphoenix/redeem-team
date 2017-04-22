#pragma once

#include <iostream>
#include <math.h>
using namespace std;

// Disable assertions
// #define NDEBUG

#define N_MOVIES 17770
#define N_USERS 458293
#define N_DAYS 2243
#define N_TRAINING 99666408
#define MAX_RATING 5
#define USER_IDX 0
#define MOVIE_IDX 1
#define TIME_IDX 2
#define RATING_IDX 3
#define DATA_POINT_SIZE 4

// Returns the differences in ms.
static double diffclock(clock_t clock1, clock_t clock2) {
    double diffticks = clock1 - clock2;
    double diffms = (diffticks) / (CLOCKS_PER_SEC / 1000);
    return diffms;
}

namespace utils {


    // Returns random value from uniform distribution.
    double uniform(double min, double max) {
        return rand() / (RAND_MAX + 1.0) * (max - min) + min;
    }

    // Returns binomial coefficient.
    int binomial(int n, double p) {
        if(p < 0 || p > 1) return 0;

        int c = 0;
        double r;

        for(int i=0; i<n; i++) {
            r = rand() / (RAND_MAX + 1.0);
            if (r < p) c++;
        }

        return c;
    }

    // Returns sigmoid of x.
    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    // bounds the prediction to between 1 and 5
    double bound(double x) {
        if (x > 5) return 5;
        if (x < 1) return 1;
        return x;
    }

    // Returns random user for minibatch
    unsigned int minibatchRandom() {
        return ((unsigned int) (rand() * 14)) % N_USERS;
    }

    // Returns positive random double between 0 and 1
    double oneRand() {
        int num = ((int) rand()) % 100;
        return num / 100.0;
    }

    // matrix subtraction of two double matrices
    // sign will be positive 1 if we're adding and -1 for subtraction
    void matrixAdd(double ** mat1, double ** mat2, unsigned int r, unsigned int c, int sign) {
        for (unsigned int i = 0; i < r; ++i) {
            for (unsigned int j = 0; j < c; ++j) {
                mat1[i][j] = mat1[i][j] - (sign * mat2[i][j]);
            }
        }
    }

    // matrix subtraction of two 3d double matrices
    // sign will be positive 1 if we're adding and -1 for subtraction
    void matrixAdd(double *** mat1, double *** mat2, unsigned int r, unsigned int c, unsigned int h, int sign) {
        for (unsigned int i = 0; i < r; ++i) {
            for (unsigned int j = 0; j < c; ++j) {
                for(unsigned int k = 0; k < h; ++k) {
                    mat1[i][j][k] = mat1[i][j][k] + (sign * mat2[i][j][k]);
                }
            }
        }
    }

    // scalar multiplication of a double 3d matrix
    void matrixScalarMult(double *** mat1, double scalar, unsigned int r, unsigned int c, unsigned int h) {
        for (unsigned int i = 0; i < r; ++i) {
            for (unsigned int j = 0; j < c; ++j) {
                for(unsigned int k = 0; k < h; ++k) {
                    mat1[i][j][k] = mat1[i][j][k] * scalar;
                }
            }
        }
    }
}
