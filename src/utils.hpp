#pragma once

#include <iostream>
#include <math.h>
using namespace std;


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

}
