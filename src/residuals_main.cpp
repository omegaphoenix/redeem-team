#include "baseline.hpp"
#include "knn.hpp"
#include "pure_rbm.hpp"

#include <assert.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>
#include <time.h>

// Takes a file of predictions and gets their residuals, storing them in
// another file
void getResiduals(std::string output_file, std::string data_file, std::string res_file) {
    printf("Getting residuals from %s against %s\n", output_file.c_str(), data_file.c_str());

    clock_t time0 = clock();
    Model *validator = new Model();
    validator->load(data_file);

    FILE* f = fopen(output_file.c_str(), "r");
    std::ofstream outputFile;
    outputFile << std::setprecision(3);
    outputFile.open(res_file);

    int i = 0;
    float prediction;
    int itemsRead = fscanf(f, "%f\n", &prediction);

    while (itemsRead == 1) {
        float actual = validator->values[i];
        float residual = actual - prediction;
        outputFile << residual << "\n";
        i++;
        itemsRead = fscanf(f, "%f\n", &prediction);
    }

    fclose(f);
    outputFile.close();

    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Getting residuals took %f ms\n", ms1);
    delete validator;
}

// Adds scores in two files, line by line
void sumFiles(std::string file_a, std::string file_b, std::string result) {
    printf("Adding predictions from %s and %s\n", file_a.c_str(), file_b.c_str());

    clock_t time0 = clock();

    FILE* a = fopen(file_a.c_str(), "r");
    FILE* b = fopen(file_b.c_str(), "r");
    std::ofstream outputFile;
    outputFile << std::setprecision(3);
    outputFile.open(result);

    int i = 0;
    float prediction_a;
    float prediction_b;
    int itemsReadA = fscanf(a, "%f\n", &prediction_a);
    int itemsReadB = fscanf(b, "%f\n", &prediction_b);

    while (itemsReadA == 1 && itemsReadB == 1) {
        float prediction = prediction_a + prediction_b;
        if (prediction < 1) {
            prediction = 1;
        }
        else if (prediction > 5) {
            prediction = 5;
        }
        outputFile << prediction << "\n";
        i++;
        itemsReadA = fscanf(a, "%f\n", &prediction_a);
        itemsReadB = fscanf(b, "%f\n", &prediction_b);
    }

    fclose(a);
    fclose(b);
    outputFile.close();

    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Summing files took %f ms\n", ms1);
    printf("Number of final predictions: %i\n", i);
}

// Validates scores in a file against given file
float validateFile(std::string pfile, std::string vfile) {
    printf("Validating predictions from %s against %s\n", pfile.c_str(), vfile.c_str());

    clock_t time0 = clock();
    Model *validator = new Model();
    validator->load(vfile);

    FILE* f = fopen(pfile.c_str(), "r");

    int i = 0;
    float prediction;
    float squareError = 0;
    int itemsRead = fscanf(f, "%f\n", &prediction);

    while (itemsRead == 1) {
        float actual = validator->values[i];
        float error = prediction - actual;
        squareError += error * error;
        assert (squareError >= 0);
        i++;
        itemsRead = fscanf(f, "%f\n", &prediction);
    }

    fclose(f);

    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Validation took %f ms\n", ms1);
    float RMSE = sqrt(squareError / validator->numRatings);
    delete validator;
    return RMSE;
}

// Train kNN on RBM residuals
int main(int argc, char **argv) {
    clock_t timestart = clock();

    std::string id = "rbm_200"; // use this to make file names unique
    std::string rbm_on_all = "model/res/rbm_predictions_all_" + id + ".out";
    std::string res_file = "model/res/rbm_residuals_alldta_" + id + ".out";
    std::string rbm_output_file = "model/res/RES_rbm_" + id + ".out";
    std::string knn_output_file = "model/res/RES_knn_" + id + ".out";
    std::string rbmknn_output_file = "out/res/RES_rbmknn_" + id + ".out";
    std::string rbm_valid_file = "model/res/RES_rbm_" + id + ".valid";
    std::string knn_valid_file = "model/res/RES_knn_" + id + ".valid";
    std::string rbmknn_valid_file = "model/res/RES_rbmknn_" + id + ".valid";

    // train RBM
    RBM* rbm = new RBM();
    rbm->init();
    rbm->loadSaved("model/rbm/pure_rbm_v3_factors_200_epoch_44_T_9.txt");
    rbm->output(rbm_on_all, "all.dta");

    // get RBM residuals
    getResiduals(rbm_on_all, "all.dta", res_file);

    // get baseline
    Baseline* base = new Baseline();
    std::string data_file = "all.dta";
    base->load(data_file);
    base->loadResiduals(res_file);
    base->train("unused variable");

    // train kNN on residuals
    kNN* knn = new kNN();
    knn->load(data_file);
    knn->loadResiduals(res_file);
    knn->transposeMU();
    knn->avg_array = base->average_array;
    knn->stdev_array = base->stdev_array;

    knn->metric = kPearson;
    knn->shared_threshold = 6;
    knn->individual_threshold = 3500;
    knn->K = 10;

    knn->train("model/knn/RES_" + knn->getFilename(res_file) + ".save");

    // validate
    rbm->output(rbm_valid_file, "4.dta");
    knn->output(knn_valid_file, "4.dta");
    sumFiles(rbm_valid_file, knn_valid_file, rbmknn_valid_file);
    float prmse = validateFile(rbmknn_valid_file, "4.dta");
    std::cout << "prmse = " << prmse << "\n";

    // predict
    rbm->output(rbm_output_file);
    knn->output(knn_output_file);
    sumFiles(rbm_output_file, knn_output_file, rbmknn_output_file);

    clock_t timeend = clock();
    float ms1 = diffclock(timeend, timestart);
    printf("Total residual training and prediction took %f ms\n", ms1);

    delete base;
    delete knn;
    delete rbm;
}

