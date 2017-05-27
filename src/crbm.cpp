#include "crbm.hpp"

#include "utils.hpp"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

CRBM::CRBM() {
    clock_t time0 = clock();
    debugPrint("Initializing...\n");

    // Initial weights
    for (int j = 0; j < N_MOVIES; ++j) {
        for (int i = 0; i < TOTAL_FEATURES; ++i) {
            for (int k = 0; k < SOFTMAX; ++k) {
                vishid[j][k][i] = normalRandom() * STD_DEV;
            }
            Dij[j][i] = normalRandom() * 0.001;
        }
    }

    // Initial biases
    for (int i = 0; i < TOTAL_FEATURES; ++i) {
        hidbiases[i] = normalRandom() * STD_DEV; // TODO: See how 0 does
    }

    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Initializing took %f ms\n", ms1);
    prevUser = -1;
}

CRBM::~CRBM() {
    delete tester;
}

void CRBM::init() {
    load("123.dta");
    tester = new Model();
    tester->load("4.dta");
    clock_t time0 = clock();
    debugPrint("Initializing visible biases...\n");

    // Init visible biases to logs of respective base rates over all users
    // TODO: Check this calculation
    unsigned int movCount[N_MOVIES] = {0};
    for (unsigned int i = 0; i < numRatings; ++i) {
        int movie = columns[i];
        assert (movie >= 0 && movie < N_MOVIES);
        int rating = values[i] - 1;
        assert (rating >= 0 && rating < MAX_RATING);
        movCount[movie] += 1;
        visbiases[movie][rating] += 1;
    }
    clock_t time1 = clock();
    for (unsigned int m = 0; m < N_MOVIES; ++m) {
        for (unsigned int k = 0; k < SOFTMAX; ++k) {
            visbiases[m][k] = log(visbiases[m][k] / movCount[m]);
        }
    }
    clock_t time2 = clock();

    float ms1 = diffclock(time1, time0);
    float ms2 = diffclock(time2, time1);
    printf("Adding visible biases took %f ms\n", ms1);
    printf("Logging biases took %f ms\n", ms2);
}

void CRBM::train(std::string saveFile) {
    std::string loadFile = "";
    // Optimize current feature
    float nrmse=2., lastRMSE = 10.;
    float prmse = 0, lastPRMSE = 0;
    loopcount=0;
    float epsilonW = EPSILONW;
    float epsilonD = EPSILOND;
    float epsilonVB = EPSILONVB;
    float epsilonHB = EPSILONHB;
    float momentum = MOMENTUM;
    ZERO(CDinc);
    ZERO(visbiasinc);
    ZERO(hidbiasinc);
    int tSteps = 1; // Set this value if you are continuing run

    std::string version = "0_full";
    debugPrint("Testing score output\n");
    std::string scoreFileName = "out/crbm/v" + version + "_scores_"
        + std::to_string(TOTAL_FEATURES) + ".txt";
    FILE *validateFile = fopen(scoreFileName.c_str(), "a");
    fprintf(validateFile, "New run\n");
    fclose(validateFile);

    loadSaved(loadFile);
    debugPrint("Testing validation\n");
    prmse = validate("4.dta");
    printf("Initial prmse is %f\n", prmse);
    debugPrint("Testing quiz/test output\n");
    output("out/crbm/crbm_v" + version + "_factors_" + std::to_string(TOTAL_FEATURES)
            + "_epoch_" + std::to_string(loopcount) + "_T_" +
            std::to_string(tSteps) + ".txt", "5-1.dta");

    // Iterate through the model while the RMSE is decreasing
    while (((nrmse < (lastRMSE - E) && prmse <= lastPRMSE) || loopcount < 14
                || (loopcount < 20 && nrmse > 0.804)) && loopcount < 80  )  {
        if (loopcount >= 10) {
            tSteps = 3 + (loopcount - 10)/5;
        }

        lastRMSE = nrmse;
        lastPRMSE = prmse;
        clock_t time0 = clock();
        loopcount++;
        int ntrain = 0;
        nrmse = 0.0;

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

            int userEnd = rowIndex[u + 1];
            int userStart = rowIndex[u];

            // For all rated movies, accumulate contributions to hidden units
            float sumW[TOTAL_FEATURES];
            ZERO(sumW);
            for (int j = userStart; j < userEnd; ++j) {
                int m = columns[j];
                moviecount[m]++;

                // 1. get one data point from data set.
                // 2. use values of this data point to set state of visible neurons Si
                int r = values[j] - 1;
                assert(r >= 0 && r < SOFTMAX);

                // Add to the bias contribution for set visible units
                posvisact[m][r] += 1.0;

                // For all hidden units h:
                for (int h = 0; h < TOTAL_FEATURES; ++h) {
                    // sum_j(W[i][j] * v[0][j]))
                    sumW[h]  += vishid[m][r][h];
                }

                // Add to hidden probabilities based on existence of a rating
                movieseencount[m]++;
                for(int h=0; h<TOTAL_FEATURES; h++) {
                    // sum_j(Dij * rij)
                    sumW[h]  += Dij[m][h];
                }
            }
            int testUserEnd = tester->rowIndex[u + 1];
            int testUserStart = tester->rowIndex[u];
            for (int j = testUserStart; j < testUserEnd; ++j) {
                int m = tester->columns[j];
                // Add to hidden probabilities based on existence of a rating
                movieseencount[m]++;
                for(int h=0; h<TOTAL_FEATURES; h++) {
                    // sum_j(Dij * rij)
                    sumW[h]  += Dij[m][h];
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
            std::copy(&poshidstates[0], &poshidstates[TOTAL_FEATURES],
                      curposhidstates);

            // Make T Contrastive Divergence steps
            int stepT = 0;
            do {
                // Determine if this is the last pass through this loop
                int finalTStep = (stepT+1 >= tSteps);

                // 5. on visible neurons compute Si using the Sj computed in step3. This is known as reconstruction
                // for all visible units j:
                int count = userEnd - userStart;
                // count += useridx[u][2];  // to compute probe errors
                // TODO: Need to add probe or validation set somehow
                for (int j = userStart; j < userEnd; ++j) {
                    int m = columns[j];
                    for (int h = 0; h < TOTAL_FEATURES; ++h) {
                        // Accumulate Weight values for sampled hidden states == 1
                        if (curposhidstates[h] == 1) {
                            for (int r = 0; r < SOFTMAX; ++r) {
                                negvisprobs[m * SOFTMAX + r] += vishid[m][r][h];
                            }
                        }

                        // Compute more accurate probabilites for RMSE reporting
                        if (stepT == 0) {
                            for (int r = 0; r < SOFTMAX; ++r)
                                nvp2[m * SOFTMAX + r] += poshidprobs[h] * vishid[m][r][h];
                        }
                    }

                    // compute P(v[1][j] = 1 | h[0]) # for binomial units, sigmoid(c[j] + sum_i(W[i][j] * h[0][i]))
                    // Softmax elements are handled individually here
                    for (int k = 0; k < SOFTMAX; ++k) {
                        negvisprobs[m * SOFTMAX + k] = 1./(1 + exp(-negvisprobs[m * SOFTMAX + k] - visbiases[m][k]));
                    }

                    // Normalize probabilities
                    float tsum  = 0.0;
                    for (int k = 0; k < SOFTMAX; ++k) {
                        tsum += negvisprobs[m * SOFTMAX + k];
                    }
                    if (tsum != 0) {
                        for (int k = 0; k < SOFTMAX; ++k) {
                            negvisprobs[m * SOFTMAX + k] /= tsum;
                        }
                    }
                    // Compute and Normalize more accurate RMSE reporting probabilities
                    if (stepT == 0) {
                        for (int k = 0; k < SOFTMAX; ++k) {
                            nvp2[m * SOFTMAX + k] = 1./(1 + exp(-nvp2[m * SOFTMAX + k] - visbiases[m][k]));
                        }
                        float tsum2  = 0.0;
                        for (int k = 0; k < SOFTMAX; ++k) {
                            tsum2 += nvp2[m * SOFTMAX + k];
                        }
                        if (tsum2 != 0) {
                            for (int k = 0; k < SOFTMAX; ++k) {
                                nvp2[m * SOFTMAX + k] /= tsum2;
                            }
                        }
                    }

                    // sample v[1][j] from P(v[1][j] = 1 | h[0])
                    float randval = randn();
                    if ((randval -= negvisprobs[m * SOFTMAX + 0]) <= 0.0 ) {
                        negvissoftmax[m] = 0;
                    }
                    else if ((randval -= negvisprobs[m * SOFTMAX + 1]) <= 0.0 ) {
                        negvissoftmax[m] = 1;
                    }
                    else if ((randval -= negvisprobs[m * SOFTMAX + 2]) <= 0.0 ) {
                        negvissoftmax[m] = 2;
                    }
                    else if ((randval -= negvisprobs[m * SOFTMAX + 3]) <= 0.0 ) {
                        negvissoftmax[m] = 3;
                    }
#ifdef NDEBUG
                    else {
                        negvissoftmax[m] = 4;
                    }
#else
                    else if ((randval -= negvisprobs[m * SOFTMAX + 4]) <= 0.01 ) {
                        negvissoftmax[m] = 4;
                    }
                    else {
                        assert (false);
                    }
#endif

                    // if in training data then train on it
                    if (true && finalTStep) {
                        negvisact[m][negvissoftmax[m]] += 1.0;
                    }
                }


                // 6. compute state of hidden neurons Sj again using Si from 5 step.
                // For all rated movies accumulate contributions to hidden units from sampled visible units
                ZERO(sumW);
                for (int j = userStart; j < userEnd; ++j) {
                    int m = columns[j];

                    // for all hidden units h:
                    for (int h = 0; h < TOTAL_FEATURES; ++h) {
                        sumW[h]  += vishid[m][negvissoftmax[m]][h];
                    }

                    // Add to hidden probabilities based on existence of a rating
                    for(int h = 0; h < TOTAL_FEATURES; ++h) {
                        // sum_j(Dij * rij)
                        sumW[h]  += Dij[m][h];
                    }
                }

                for (int j = testUserStart; j < testUserEnd; ++j) {
                    int m = tester->columns[j];
                    // Add to hidden probabilities based on existence of a rating
                    for(int h = 0; h < TOTAL_FEATURES; ++h) {
                        // sum_j(Dij * rij)
                        sumW[h]  += Dij[m][h];
                    }
                }

                // for all hidden units h:
                for (int h = 0; h < TOTAL_FEATURES; ++h) {
                    // compute Q(h[1][i] = 1 | v[1]) # for binomial units, sigmoid(b[i] + sum_j(W[i][j] * v[1][j]))
                    neghidprobs[h]  = 1./(1 + exp(-sumW[h] - hidbiases[h]));

                    // Sample the hidden units state again.
                    if  (neghidprobs[h] >  randn()) {
                        neghidstates[h]=1;
                        if (finalTStep) {
                            neghidact[h] += 1.0;
                        }
                    }
                    else {
                        neghidstates[h]=0;
                    }
                }

                // Compute error rmse and prmse before we start iterating on T
                if (stepT == 0) {

                    // Compute rmse on training data
                    for (int j = userStart; j < userEnd; ++j) {
                        int m = columns[j];
                        int r = values[j] - 1;
                        assert(r >= 0 && r < SOFTMAX);

                        //# Compute some error function like sum of squared difference between Si in 1) and Si in 5)
                        float expectedV = nvp2[m * SOFTMAX + 1] + 2.0 * nvp2[m * SOFTMAX + 2] + 3.0 * nvp2[m * SOFTMAX + 3] + 4.0 * nvp2[m * SOFTMAX + 4];
                        float vdelta = (((float)r)-expectedV);
                        nrmse += (vdelta * vdelta);
                    }
                    ntrain += count;
                }

                // If looping again, load the curposvisstates
                if (!finalTStep) {
                    std::copy(&neghidstates[0], &neghidstates[TOTAL_FEATURES],
                              curposhidstates);
                    ZERO(negvisprobs);
                }

                // 8. repeating multiple times steps 5,6 and 7 compute (Si.Sj)n. Where n is small number and can
                //    increase with learning steps to achieve better accuracy.

            } while (++stepT < tSteps);

            // Accumulate contrastive divergence contributions for (Si.Sj)0 and (Si.Sj)T
            for (int j = userStart; j < userEnd; ++j) {
                int m = columns[j];
                int r = values[j] - 1;
                assert(r >= 0 && r < SOFTMAX);

                // for all hidden units h:
                for (int h = 0; h < TOTAL_FEATURES; ++h) {
                    if (poshidstates[h] == 1) {
                        // 4. now Si and Sj values can be used to compute (Si.Sj)0  here () means just values not average
                        // accumulate CDpos = CDpos + (Si.Sj)0
                        CDpos[m][h][r] += 1.0;
                    }

                    // 7. now use Si and Sj to compute (Si.Sj)1 (fig.3)
                    CDneg[m][h][negvissoftmax[m]]+= (float) neghidstates[h];
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
                        for (int rr = 0; rr < SOFTMAX; ++rr) {
                            //# At the end compute average of CDpos and CDneg by dividing them by number of data points.
                            //# Compute CD = < Si.Sj >0  < Si.Sj >n = CDpos  CDneg
                            float CDp = CDpos[m][h][rr];
                            float CDn = CDneg[m][h][rr];
                            if (CDp != 0.0 || CDn != 0.0) {
                                CDp /= ((float)moviecount[m]);
                                CDn /= ((float)moviecount[m]);

                                // W += epsilon * (h[0] * v[0]' - Q(h[1][.] = 1 | v[1]) * v[1]')
                                //# Update weights and biases W = W + alpha*CD (biases are just weights to neurons that stay always 1.0)
                                //e.g between data and reconstruction.
                                CDinc[m][h][rr] = momentum * CDinc[m][h][rr] + epsilonW * ((CDp - CDn) - WEIGHTCOST * vishid[m][rr][h]);
                                vishid[m][rr][h] += CDinc[m][h][rr];
                            }
                        }
                    }

                    // Update visible softmax biases
                    // c += epsilon * (v[0] - v[1])$
                    // for all softmax
                    for (int rr = 0; rr < SOFTMAX; rr++) {
                        if (posvisact[m][rr] != 0.0 || negvisact[m][rr] != 0.0) {
                            posvisact[m][rr] /= ((float)moviecount[m]);
                            negvisact[m][rr] /= ((float)moviecount[m]);
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
                        poshidact[h]  /= ((float)(numcases));
                        neghidact[h]  /= ((float)(numcases));
                        hidbiasinc[h] = momentum * hidbiasinc[h] + epsilonHB * ((poshidact[h] - neghidact[h]));
                        //hidbiasinc[h] = momentum * hidbiasinc[h] + epsilonHB * ((poshidact[h] - neghidact[h]) - WEIGHTCOST * hidbiases[h]);
                        hidbiases[h]  += hidbiasinc[h];
                    }
                }
                // Update all DIJ factors
                for(int m = 0; m < N_MOVIES; ++m) {
                    if (movieseencount[m] == 0) {
                        continue;   // This seems correct given what I'm doing for training on rated movies
                    }
                    // for all hidden units h:
                    for(int h=0;h<TOTAL_FEATURES;h++) {
                        // Update conditional Dij factors
                        DIJinc[m][h] = momentum * DIJinc[m][h] + epsilonD * ((poshidact[h] - neghidact[h]) /*- weightcost * Dij[m][h]*/);
                        Dij[m][h]   += DIJinc[m][h];
                    }
                }
                ZERO(CDpos);
                ZERO(CDneg);
                ZERO(poshidact);
                ZERO(neghidact);
                ZERO(posvisact);
                ZERO(negvisact);
                ZERO(moviecount);
                ZERO(movieseencount);
            }
        }

        printf("nrmse: %f \n", nrmse);
        printf("ntrain: %d \n", ntrain);
        nrmse = sqrt(nrmse / ntrain);
        printf("nrmse: %f \n", nrmse);
        if (loopcount % 5 == 0 || loopcount > 40 || loopcount == 1) {
            prmse = validate("4.dta");
        }

        clock_t time1 = clock();
        float ms1 = diffclock(time1, time0);
        validateFile = fopen(scoreFileName.c_str(), "a");
        printf("epoch: %d nrmse: %f prmse: %f time: %f ms\n", loopcount, nrmse, prmse, ms1);
        fprintf(validateFile, "epoch: %d nrmse: %f prmse: %f time: %f ms\n", loopcount, nrmse, prmse, ms1);
        fclose(validateFile);
        save("model/crbm/crbm_v" + version + "_factors_" + std::to_string(TOTAL_FEATURES)
                + "_epoch_" + std::to_string(loopcount) + "_T_" +
                std::to_string(tSteps) + ".mod");
        if (loopcount % 5 == 0 || loopcount > 40 || loopcount == 1) {
            output("out/crbm/crbm_v" + version + "_factors_" + std::to_string(TOTAL_FEATURES)
                    + "_epoch_" + std::to_string(loopcount) + "_T_" +
                    std::to_string(tSteps) + ".txt", "5-1.dta");
            output("out/crbm/crbm_v" + version + "_factors_" + std::to_string(TOTAL_FEATURES)
                    + "_epoch_" + std::to_string(loopcount) + "_T_" +
                    std::to_string(tSteps) + "_4.txt", "4.dta");
            output("out/crbm/crbm_v" + version + "_factors_" + std::to_string(TOTAL_FEATURES)
                    + "_epoch_" + std::to_string(loopcount) + "_T_" +
                    std::to_string(tSteps) + "all.txt", "all.dta");
        }


        if ( loopcount > 10 ) {
            epsilonW  *= 0.91;
            epsilonD  *= 0.91;
            epsilonVB *= 0.91;
            epsilonHB *= 0.91;
        } else if ( loopcount > 9 ) {
            epsilonW  *= 0.75;
            epsilonD  *= 0.75;
            epsilonVB *= 0.75;
            epsilonHB *= 0.75;
        } else if ( loopcount > 5 ) {
            epsilonW  *= 0.91;
            epsilonD  *= 0.91;
            epsilonVB *= 0.91;
            epsilonHB *= 0.91;
        } else if ( loopcount > 4 ) {
            epsilonW  *= 0.80;
            epsilonD  *= 0.80;
            epsilonVB *= 0.80;
            epsilonHB *= 0.80;
        }
    }
    output("out/crbm/crbm_v" + version + "_factors_" + std::to_string(TOTAL_FEATURES)
            + "_epoch_" + std::to_string(loopcount) + "_T_" +
            std::to_string(tSteps) + ".txt", "5-1.dta");
    output("out/crbm/crbm_v" + version + "_factors_" + std::to_string(TOTAL_FEATURES)
            + "_epoch_" + std::to_string(loopcount) + "_T_" +
            std::to_string(tSteps) + "_4.txt", "4.dta");
    output("out/crbm/crbm_v" + version + "_factors_" + std::to_string(TOTAL_FEATURES)
            + "_epoch_" + std::to_string(loopcount) + "_T_" +
            std::to_string(tSteps) + "_all.txt", "all.dta");
}

void CRBM::prepPredict(Model *mod, int n) {
    ZERO(negvisprobs);
    int userEnd = rowIndex[n + 1];
    int userStart = rowIndex[n];
    float sumW[TOTAL_FEATURES];
    ZERO(sumW);
    for (int j = userStart; j < userEnd; ++j) {
        int m =columns[j];
        int r = values[j] - 1;
        assert(r >= 0 && r < SOFTMAX);

        for (int h = 0; h < TOTAL_FEATURES; ++h) {
            sumW[h] += vishid[m][r][h];
        }

        // Add to hidden probabilities based on existence of a rating
        for(int h = 0; h < TOTAL_FEATURES; ++h) {
            // sum_j(Dij * rij)
            sumW[h]  += Dij[m][h];
        }
    }
    int testUserEnd = mod->rowIndex[n + 1];
    int testUserStart = mod->rowIndex[n];
    for (int j = testUserStart; j < testUserEnd; ++j) {
        int m = mod->columns[j];
        // Add to hidden probabilities based on existence of a rating
        for(int h = 0; h < TOTAL_FEATURES; ++h) {
            // sum_j(Dij * rij)
            sumW[h]  += Dij[m][h];
        }
    }

    // Compute hidden probabilities
    for (int h = 0; h < TOTAL_FEATURES; ++h) {
        poshidprobs[h] = 1.0 / (1.0 + exp(-sumW[h] - hidbiases[h]));
    }

    for (int j = testUserStart; j < testUserEnd; ++j) {
        int m = mod->columns[j];
        for (int h = 0; h < TOTAL_FEATURES; ++h) {
            for (int k = 0; k < SOFTMAX; ++k) {
                negvisprobs[m * SOFTMAX + k] += poshidprobs[h] * vishid[m][k][h];
            }
        }

        for (int k = 0; k < SOFTMAX; ++k) {
            negvisprobs[m * SOFTMAX + k]  = 1./(1 + exp(-negvisprobs[m * SOFTMAX + k] - visbiases[m][k]));
        }

        float tsum = 0.0;
        for (int k = 0; k < SOFTMAX; ++k) {
            tsum += negvisprobs[m * SOFTMAX + k];
        }

        if (tsum != 0) {
            for (int k = 0; k < SOFTMAX; ++k) {
                negvisprobs[m * SOFTMAX + k] /= tsum;
            }
        }
        else {
            negvisprobs[m * SOFTMAX + 2] = 1.0;
        }
    }
    prevUser = n;
}

// Return the predicted rating for user n, movie i
float CRBM::predict(int n, int i, int d) {
    assert (n == prevUser);
    float expVal = 0.0;
    for (int k = 0; k < SOFTMAX; ++k) {
        expVal += negvisprobs[i * SOFTMAX + k] * (k + 1);
    }
    assert(expVal >= 1 && expVal <= 5);
    return expVal;
}

// Use <stdio.h> for binary writing.
void CRBM::save(std::string fname) {
    clock_t time0 = clock();
    debugPrint("Saving...\n");

    FILE *out = fopen(fname.c_str(), "wb");
    int buf[1];
    buf[0] = loopcount;
    fwrite(buf, sizeof(int), 1, out);
    fwrite(vishid, sizeof(float), N_MOVIES * SOFTMAX * TOTAL_FEATURES, out);
    fwrite(visbiases, sizeof(float), N_MOVIES * SOFTMAX, out);
    fwrite(visbiasinc, sizeof(float), N_MOVIES * SOFTMAX, out);
    fwrite(hidbiases, sizeof(float), TOTAL_FEATURES, out);
    fwrite(hidbiasinc, sizeof(float), TOTAL_FEATURES, out);
    fwrite(CDinc, sizeof(float), N_MOVIES * TOTAL_FEATURES * SOFTMAX, out);
    fwrite(Dij, sizeof(float), N_MOVIES * TOTAL_FEATURES, out);
    fwrite(DIJinc, sizeof(float), N_MOVIES * TOTAL_FEATURES, out);
    fclose(out);

    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Saving took %f ms\n", ms1);
}

void CRBM::loadSaved(std::string fname) {
    clock_t time0 = clock();
    debugPrint("Loading saved...\n");

    FILE *in = fopen(fname.c_str(), "r");
    if (fname == "" || in == NULL) {
    }
    else {
        debugPrint("Loading saved CRBM...\n");
        // Buffer to hold numEpochs
        int buf[1];
        fread(buf, sizeof(int), 1, in);
        loopcount = buf[0];

        // Initialize vishid, visbiases, hidbiases
        fread(vishid, sizeof(float), N_MOVIES * SOFTMAX * TOTAL_FEATURES, in);
        fread(visbiases, sizeof(float), N_MOVIES * SOFTMAX, in);
        fread(visbiasinc, sizeof(float), N_MOVIES * SOFTMAX, in);
        fread(hidbiases, sizeof(float), TOTAL_FEATURES, in);
        fread(hidbiasinc, sizeof(float), TOTAL_FEATURES, in);
        fread(CDinc, sizeof(float), N_MOVIES * TOTAL_FEATURES * SOFTMAX, in);
        fread(Dij, sizeof(float), N_MOVIES * TOTAL_FEATURES, in);
        fread(DIJinc, sizeof(float), N_MOVIES * TOTAL_FEATURES, in);
        fclose(in);
    }

    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Loading saved took %f ms\n", ms1);
}

int main() {
    // Speed up stdio operations
    std::ios_base::sync_with_stdio(false);

    srand(0);

    clock_t time0 = clock();
    // Initialize
    CRBM* crbm = new CRBM();
    clock_t time1 = clock();
    crbm->init();
    clock_t time2 = clock();

    // Learn parameters
    crbm->train("unused");
    clock_t time3 = clock();

    float ms1 = diffclock(time1, time0);
    float ms2 = diffclock(time2, time1);
    float ms3 = diffclock(time3, time2);
    float total_ms = diffclock(time3, time0);

    printf("Initialization took %f ms\n", ms1);
    printf("Total loading took %f ms\n", ms2);
    printf("Training took %f ms\n", ms3);
    printf("Total running time was %f ms\n", total_ms);

    delete crbm;
    return 0;
}
