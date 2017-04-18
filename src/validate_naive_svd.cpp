#include <algorithm>
#include <iostream>
#include <string>
#include "naive_svd.hpp"

// Indicates whether to train
#define TRAIN true
#define INIT_ERROR 10

int main(int argc, char** argv) {

    std::vector<int> ks = {40, 50, 60};
    std::vector<float> lambdas = {0.0001, 0.001, 0.01, 0.1, 1.0};

    NaiveSVD* nsvd = new NaiveSVD();
    nsvd->load("1.dta");
    float error = INIT_ERROR;
    std::string best_model;
    int best_k;
    float best_lambda;
    for (int i = 0; i < ks.size(); i++) {
        for (int j = 0; j < lambdas.size(); j++) {
            int k = ks[i];
            float lamb = lambdas[j];

            std::string fname = "model/naive_svd/k=" + std::to_string(k) +
                                "_lamb=" + std::to_string(lamb) + "_epoch=" +
                                std::to_string(nsvd->numEpochs) + ".save";
            nsvd->setParams(k, 0.001, lamb);
            #ifdef TRAIN
                nsvd->train(fname);
            #else
                nsvd->loadSaved(fname);
            #endif

            for (int epoch = 10; epoch <= nsvd->numEpochs; epoch += 10) {
                fname = "model/naive_svd/k=" + std::to_string(k) + "_lamb=" +
                        std::to_string(lamb) + "_epoch=" + std::to_string(epoch) + ".save";
                float cur_error = nsvd->validate("2.dta", fname);
                std::cerr << "Error for " << fname << std::endl <<
                          cur_error << std::endl;
                if (cur_error < error) {
                    error = cur_error;
                    best_model = fname;
                    best_k = k;
                    best_lambda = lamb;
                }
            }
        }
    }
    nsvd->setParams(best_k, 0.001, best_lambda);
    nsvd->loadSaved(best_model);
    nsvd->printOutput("out/naive_svd_validation.dta");

    return 0;
}
