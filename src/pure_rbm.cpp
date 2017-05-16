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

        for (int u = 0; u < N_USERS; ++u) {

            // Clear summations for probabilities
            ZERO(negvisprobs);
            ZERO(nvp2);

            // perform steps 1 to 8
            int userEnd = rowIndex[u + 1];
            int userStart = rowIndex[u];

            // For all rated movies, accumulate contributions to hidden units
            double sumW[TOTAL_FEATURES];
            ZERO(sumW);
            for (int j = userStart; j < userEnd; ++j) {
                int m = columns[j];
                moviecount[m]++;

                // 1. get one data point from data set.
                // 2. use values of this data point to set state of visible neurons Si
                int r = values[j];

                // Add to the bias contribution for set visible units
                posvisact[m][r] += 1.0;

                // For all hidden units h:
                for (int h = 0; h < TOTAL_FEATURES; ++h) {
                    // sum_j(W[i][j] * v[0][j]))
                    sumW[h]  += vishid[m][r][h];
                }
            }

            // Sample the hidden units state after computing probabilities
            for (int h = 0; h < TOTAL_FEATURES; ++h) {

                // 3. compute Sj for each hidden neuron based on formula above and states of visible neurons Si
                // poshidprobs[h] = 1./(1 + exp(-V*vishid - hidbiases);
                // compute Q(h[0][i] = 1 | v[0]) # for binomial units, sigmoid(b[i] + sum_j(W[i][j] * v[0][j]))
                poshidprobs[h]  = 1.0 / (1.0 + exp(-sumW[h] - hidbiases[h]));

                // sample h[0][i] from Q(h[0][i] = 1 | v[0])
                if  (poshidprobs[h] >  randn()) {
                    poshidstates[h]=1;
                    poshidact[h] += 1.0;
                } else {
                    poshidstates[h]=0;
                }
            }

            // Load up a copy of poshidstates for use in loop
            for (int h = 0; h < TOTAL_FEATURES; ++h) {
                curposhidstates[h] = poshidstates[h];
            }

            // Make T Contrastive Divergence steps
            int stepT = 0;

            // Accumulate contrastive divergence contributions for (Si.Sj)0 and (Si.Sj)T
            for (int j = userStart; j < userEnd; ++j) {
                int m = columns[j];
                int r = values[j];

                // for all hidden units h:
                for (int h = 0; h < TOTAL_FEATURES; ++h) {
                    if (poshidstates[h] == 1) {
                        // 4. now Si and Sj values can be used to compute (Si.Sj)0  here () means just values not average
                        // accumulate CDpos = CDpos + (Si.Sj)0
                        CDpos[m][r][h] += 1.0;
                    }

                    // 7. now use Si and Sj to compute (Si.Sj)1 (fig.3)
                    CDneg[m][negvissoftmax[m]][h] += (double) neghidstates[h];
                }
            }

            // Update weights and biases after batch
            //
            int bsize = BATCH_SIZE;
            if (((u+1) % bsize) == 0 || (u+1) == N_USERS) {
                int numcases = u % bsize;
                numcases++;

                // Update weights
                for (int m = 0 ; m < N_MOVIES; ++m) {
                    if ( moviecount[m] == 0 ) {
                        continue;
                    }

                    // for all hidden units h:
                    for (int h = 0; h < TOTAL_FEATURES; ++h) {
                        // for all softmax
                        for (int rr = 0; rr < SOFTMAX; rr++) {
                            //# At the end compute average of CDpos and CDneg by dividing them by number of data points.
                            //# Compute CD = < Si.Sj >0  < Si.Sj >n = CDpos  CDneg
                            double CDp = CDpos[m][rr][h];
                            double CDn = CDneg[m][rr][h];
                            if (CDp != 0.0 || CDn != 0.0) {
                                CDp /= ((double)moviecount[m]);
                                CDn /= ((double)moviecount[m]);

                                // W += epsilon * (h[0] * v[0]' - Q(h[1][.] = 1 | v[1]) * v[1]')
                                //# Update weights and biases W = W + alpha*CD (biases are just weights to neurons that stay always 1.0)
                                //e.g between data and reconstruction.
                                CDinc[m][rr][h] = momentum * CDinc[m][rr][h] + epsilonW * ((CDp - CDn) - WEIGHTCOST * vishid[m][rr][h]);
                                vishid[m][rr][h] += CDinc[m][rr][h];
                            }
                        }
                    }

                    // Update visible softmax biases
                    // c += epsilon * (v[0] - v[1])$
                    // for all softmax
                    for (int rr = 0; rr < SOFTMAX; rr++) {
                        if (posvisact[m][rr] != 0.0 || negvisact[m][rr] != 0.0) {
                            posvisact[m][rr] /= ((double)moviecount[m]);
                            negvisact[m][rr] /= ((double)moviecount[m]);
                            visbiasinc[m][rr] = momentum * visbiasinc[m][rr] + epsilonVB * ((posvisact[m][rr] - negvisact[m][rr]));
                            //visbiasinc[m][rr] = momentum * visbiasinc[m][rr] + epsilonVB * ((posvisact[m][rr] - negvisact[m][rr]) - WEIGHTCOST * visbiases[m][rr]);
                            visbiases[m][rr]  += visbiasinc[m][rr];
                        }
                    }
                }


                // Update hidden biases
                // b += epsilon * (h[0] - Q(h[1][.] = 1 | v[1]))
                for (int h = 0; h < TOTAL_FEATURES; ++h) {
                    if (poshidact[h]  != 0.0 || neghidact[h]  != 0.0) {
                        poshidact[h]  /= ((double)(numcases));
                        neghidact[h]  /= ((double)(numcases));
                        hidbiasinc[h] = momentum * hidbiasinc[h] + epsilonHB * ((poshidact[h] - neghidact[h]));
                        //hidbiasinc[h] = momentum * hidbiasinc[h] + epsilonHB * ((poshidact[h] - neghidact[h]) - WEIGHTCOST * hidbiases[h]);
                        hidbiases[h]  += hidbiasinc[h];
                    }
                }
                ZERO(CDpos);
                ZERO(CDneg);
                ZERO(poshidact);
                ZERO(neghidact);
                ZERO(posvisact);
                ZERO(negvisact);
                ZERO(moviecount);
            }
        }

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

// Return the predicted rating for user n, movie i
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
