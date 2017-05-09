#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <math.h>

using namespace std;

// Disable assertions
// #define NDEBUG
// Uncomment this if you want to print debug statements
#define PRINT

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

static inline void debugPrint(const char* statement) {
#ifdef PRINT
    printf("%s", statement);
#endif
}

// Returns a uniformly distributed random number
static inline double uniformRandom() {
  return ( (double)(rand()) + 1. )/( (double)(RAND_MAX) + 1. );
}

// Returns a normally distributed random number
static inline double normalRandom() {
  double u1=uniformRandom();
  double u2=uniformRandom();
  return cos(8.*atan(1.)*u2)*sqrt(-2.*log(u1));
}

// Returns the differences in ms.
static inline double diffclock(clock_t clock1, clock_t clock2) {
    double diffticks = clock1 - clock2;
    double diffms = (diffticks) / (CLOCKS_PER_SEC / 1000);
    return diffms;
}

// Returns random value from uniform distribution.
static inline double uniform(double min, double max) {
    return rand() / (RAND_MAX + 1.0) * (max - min) + min;
}

// Returns binomial coefficient.
static inline int binomial(int n, double p) {
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
static inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// bounds the prediction to between 1 and 5
static inline double bound(double x) {
    if (x > 5) return 5;
    if (x < 1) return 1;
    return x;
}

static unsigned long x=123456789, y=362436069, z=521288629;
static inline unsigned int minibatchRandom(void) {
    //period 2^96-1
    unsigned long t;
    x ^= x << 16;
    x ^= x >> 5;
    x ^= x << 1;

    t = x;
    x = y;
    y = z;
    z = t ^ x ^ y;

    return (unsigned int) (z % N_USERS);
}

// Returns positive random double between 0 and 1
static inline double oneRand() {
    int num = ((int) rand()) % 100;
    return num / 100.0;
}

// Matrix subtraction of two double matrices
// Sign will be positive 1 if we're adding and -1 for subtraction
static inline void matrixAdd(double ** mat1, double ** mat2, unsigned int r, unsigned int c, int sign) {
    for (unsigned int i = 0; i < r; ++i) {
        for (unsigned int j = 0; j < c; ++j) {
            mat1[i][j] = mat1[i][j] - (sign * mat2[i][j]);
        }
    }
}

// Matrix subtraction of two 3d double matrices
// Sign will be positive 1 if we're adding and -1 for subtraction
static inline void matrixAdd(double *** mat1, double *** mat2, unsigned int r, unsigned int c, unsigned int h, int sign) {
    for (unsigned int i = 0; i < r; ++i) {
        for (unsigned int j = 0; j < c; ++j) {
            for(unsigned int k = 0; k < h; ++k) {
                mat1[i][j][k] = mat1[i][j][k] + (sign * mat2[i][j][k]);
            }
        }
    }
}

// Scalar multiplication of a double 3d matrix
static inline void matrixScalarMult(double *** mat1, double scalar, unsigned int r, unsigned int c, unsigned int h) {
    for (unsigned int i = 0; i < r; ++i) {
        for (unsigned int j = 0; j < c; ++j) {
            for(unsigned int k = 0; k < h; ++k) {
                mat1[i][j][k] = mat1[i][j][k] * scalar;
            }
        }
    }
}
#endif // UTILS_HPP
