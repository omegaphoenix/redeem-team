#include "pure_rbm.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

RBM::RBM() {
}

RBM::~RBM() {
}

void RBM::init() {
    load("1.dta");
    clock_t time0 = clock();
    debugPrint("Initializing...\n");

    // Initial weights
    for (int j = 0; j < N_MOVIES; ++j) {
        for (int i = 0; i < TOTAL_FEATURES; ++i) {
            for (int k = 0; k < SOFTMAX; ++k) {
                // TODO: Normal Distribution
                vishid[j][k][i] = 0.02 * randn() - 0.004;
            }
        }
    }

    // Initial biases
    for (int i = 0; i < TOTAL_FEATURES; ++i) {
        hidbiases[i] = 0.0;
    }

    for (int j=0; j < N_MOVIES; ++j) {
        for (int i = 0; i < SOFTMAX; ++i) {
            // TODO: Normal Distribution
            visbiases[j][i] = 0.02 * randn() - 0.004;
        }
    }
    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Initializing took %f ms\n", ms1);
}

void RBM::train(std::string saveFile) {
    init();

    // Optimize current feature
    double nrmse=2., lastRMSE=10.;
    double prmse = 0, lastPRMSE=0;
    int loopcount=0;
    double epsilonW = EPSILONW;
    double epsilonVB = EPSILONVB;
    double epsilonHB = EPSILONHB;
    double momentum = MOMENTUM;
    ZERO(CDinc);
    ZERO(visbiasinc);
    ZERO(hidbiasinc);
    int tSteps = 1;

    // Iterate through the model while the RMSE is decreasing
    while (((nrmse < (lastRMSE-E)) || loopcount < 14) && loopcount < 80)  {
        if ( loopcount >= 10 ) {
            tSteps = 3 + (loopcount - 10)/5;
        }

        lastRMSE = nrmse;
        lastPRMSE = prmse;
        clock_t time0 = clock();
        loopcount++;
        int ntrain = 0;
        nrmse = 0.0;
        double s  = 0.0;
        int n = 0;

        if (loopcount > 5) {
            momentum = FINAL_MOMENTUM;
        }

        // CDpos =0, CDneg=0 (matrices)
        ZERO(CDpos);
        ZERO(CDneg);
        ZERO(poshidact);
        ZERO(neghidact);
        ZERO(posvisact);
        ZERO(negvisact);
        ZERO(moviecount);


        nrmse = sqrt(nrmse / ntrain);
        prmse = sqrt(s / n);

        clock_t time1 = clock();
        float ms1 = diffclock(time1, time0);
        printf("nrmse: %f\t prmse: %f time: %f ms\n", nrmse, prmse, ms1);

        if (TOTAL_FEATURES == 200) {
            if (loopcount > 6) {
                epsilonW  *= 0.90;
                epsilonVB *= 0.90;
                epsilonHB *= 0.90;
            } else if (loopcount > 5) {  // With 200 hidden variables, you need to slow things down a little more
                epsilonW  *= 0.50; // This could probably use some more optimization
                epsilonVB *= 0.50;
                epsilonHB *= 0.50;
            } else if (loopcount > 2) {
                epsilonW  *= 0.70;
                epsilonVB *= 0.70;
                epsilonHB *= 0.70;
            }
        } else {  // The 100 hidden variable case
            if (loopcount > 8) {
                epsilonW  *= 0.92;
                epsilonVB *= 0.92;
                epsilonHB *= 0.92;
            } else if (loopcount > 6) {
                epsilonW  *= 0.90;
                epsilonVB *= 0.90;
                epsilonHB *= 0.90;
            } else if (loopcount > 2) {
                epsilonW  *= 0.78;
                epsilonVB *= 0.78;
                epsilonHB *= 0.78;
            }
        }
    }
}

float RBM::predict(int n, int i) {
    return 0;
}

int main() {
    // Speed up stdio operations
    std::ios_base::sync_with_stdio(false);

    srand(0);

    clock_t time0 = clock();
    // Initialize
    RBM* rbm = new RBM();
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
