#include <assert.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "Eigen/Dense"

#define N_QUIZ 1374949
// N_QUAL: 2749898
#define QUIZ_SQUARE 3.84358

using namespace std;
using namespace Eigen;

// naive_svd.dta : 0.95967
const vector<string> models = {"naive_svd_validation.dta", "timesvdpp/50factors_30bins_2epochs.out", "naive_svd.dta"};
const vector<double> rmses = {0.9432, 0.93826, 0.95967};


VectorXd getPrediction(string fname) {
    VectorXd prediction(N_QUIZ);
    fname = "out/" + fname;
    std::ifstream f(fname.c_str());
    if (!f.good()) {
        printf("%s is missing\n", fname.c_str());
        assert (f.good());
    }
    for (int i = 0; i < N_QUIZ; ++i) {
        double cur;
        f >> cur;
        prediction(i) = cur;
    }
    f.close();
    return prediction;
}

int main(int argc, char** argv) {
    int N_MODELS = models.size();
    vector<VectorXd> predictions(N_MODELS);
    MatrixXd A(N_QUIZ, N_MODELS);
    for (int i = 0; i < N_MODELS; ++i) {
        predictions[i] = getPrediction(models[i]);
        A.col(i) = predictions[i];
    }
    VectorXd aTs(N_MODELS);
    VectorXd alpha(N_MODELS);

    for (int i = 0; i < N_MODELS; ++i) {
        aTs(i) = 0.5 * (predictions[i].dot(predictions[i]) +
                 N_QUIZ * QUIZ_SQUARE * QUIZ_SQUARE - N_QUIZ * rmses[i] * rmses[i]);
    }

    alpha = (A.transpose() * A).inverse() * aTs;
    cout << "Alpha: \n" << alpha << "\n";

    VectorXd newPredict(N_QUIZ);
    newPredict = A * alpha;

    std::string outFname = "out/blend/blend_4_tsvdpp_2_crbm_2_rbm_missing"; // leave off the .txt
    std::string noisyOutFname = outFname + "_noisy.txt";
    outFname = outFname + ".txt";
    ofstream outputFile, noisyOutputFile;
    outputFile.open(outFname);
    noisyOutputFile.open(noisyOutFname);
    outputFile << setprecision(3);
    noisyOutputFile << setprecision(3);

    bool noiseFlag = true;
    float noise = 0.25;
    int count = 0;
    for (int i = 0; i < N_QUIZ; ++i) {
        float rating = newPredict(i);
        if (rating < 1) {
            rating = 1;
        }
        else if (rating > 5) {
            rating = 5;
        }
        // Use \n for efficiency.
        outputFile << rating << "\n";

        if (noiseFlag) {
            rating += noise;
            if (rating <= 5.) {
                noiseFlag = false;
                count++;
            }
            else {
                rating -= noise;
            }
        }
        else {
            rating -= noise;
            if (rating >= 1.) {
                noiseFlag = true;
                count++;
            }
            else {
                rating += noise;
            }
        }
        noisyOutputFile << rating << "\n";
    }
    outputFile.close();
    noisyOutputFile.close();
    printf("Points modified: %d\n", count);

    return 0;
}
