#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <math.h>

using namespace std;

// Uncomment to disable assertions
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

struct dataPoint {
    int userID;
    int movieID;
    int date;
    int value;

    dataPoint() {
        userID = 0;
        movieID = 0;
        date = 0;
        value = 0;
    }

    dataPoint(int a, int b, int c, int d) :
        userID(a), movieID(b), date(c), value(d) {
    }

    // Sort my movie
    bool operator<(const struct dataPoint &other) const
    {
        if (movieID != other.movieID) {
            return movieID < other.movieID;
        }
        else {
            return userID < other.userID;
        }
    }
};


static inline void debugPrint(const char* statement) {
#ifdef PRINT
    printf("%s", statement);
#endif
}

double randn() {
    return (rand()/(double)(RAND_MAX));
}

// Returns a uniformly distributed random number
static inline float uniformRandom() {
  return ( (float)(rand()) + 1. )/( (float)(RAND_MAX) + 1. );
}

// Returns a normally distributed random number
static inline float normalRandom() {
  float u1=uniformRandom();
  float u2=uniformRandom();
  return cos(8.*atan(1.)*u2)*sqrt(-2.*log(u1));
}

// Returns the differences in ms.
static inline float diffclock(clock_t clock1, clock_t clock2) {
    float diffticks = clock1 - clock2;
    float diffms = (diffticks) / (CLOCKS_PER_SEC / 1000);
    return diffms;
}

// Returns random value from uniform distribution.
static inline float uniform(float min, float max) {
    return rand() / (RAND_MAX + 1.0) * (max - min) + min;
}

// Returns binomial coefficient.
static inline int binomial(int n, float p) {
    if(p < 0 || p > 1) return 0;

    int c = 0;
    float r;

    for(int i=0; i<n; i++) {
        r = rand() / (RAND_MAX + 1.0);
        if (r < p) c++;
    }

    return c;
}

// Returns sigmoid of x.
static inline float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-1.0 * x));
}

// bounds the prediction to between 1 and 5
static inline float bound(float x) {
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

// Returns positive random float between 0 and 1
static inline float oneRand() {
    int num = ((int) rand()) % 100;
    return num / 100.0;
}

// Matrix subtraction of two float matrices
// Sign will be positive 1 if we're adding and -1 for subtraction
static inline void matrixAdd(float ** mat1, float ** mat2, unsigned int r, unsigned int c, int sign) {
    for (unsigned int i = 0; i < r; ++i) {
        for (unsigned int j = 0; j < c; ++j) {
            mat1[i][j] = mat1[i][j] - (sign * mat2[i][j]);
        }
    }
}

// Matrix subtraction of two 3d float matrices
// Sign will be positive 1 if we're adding and -1 for subtraction
static inline void matrixAdd(float *** mat1, float *** mat2, unsigned int r, unsigned int c, unsigned int h, int sign) {
    for (unsigned int i = 0; i < r; ++i) {
        for (unsigned int j = 0; j < c; ++j) {
            for(unsigned int k = 0; k < h; ++k) {
                mat1[i][j][k] = mat1[i][j][k] + (sign * mat2[i][j][k]);
            }
        }
    }
}

// Scalar multiplication of a float 3d matrix
static inline void matrixScalarMult(float *** mat1, float scalar, unsigned int r, unsigned int c, unsigned int h) {
    for (unsigned int i = 0; i < r; ++i) {
        for (unsigned int j = 0; j < c; ++j) {
            for(unsigned int k = 0; k < h; ++k) {
                mat1[i][j][k] = mat1[i][j][k] * scalar;
            }
        }
    }
}
#endif // UTILS_HPP
