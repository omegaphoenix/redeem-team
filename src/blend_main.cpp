#include <assert.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "Eigen/Dense"

#define N_QUIZ 2749898
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
    assert (f.good());
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
                 QUIZ_SQUARE - rmses[i]);
    }

    alpha = (A.transpose() * A).inverse() * aTs;
    cout << "Alpha: " << alpha << "\n";

    VectorXd newPredict(N_QUIZ);
    newPredict = A * alpha;

    ofstream outputFile;
    outputFile.open("out/blend/blend.dta");
    outputFile << setprecision(3);
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
    }
    outputFile.close();

    return 0;
}
