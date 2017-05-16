#include <iostream>
#include <time.h>
#include "svd_plusplus.hpp"

int main(int argc, char **argv) {
    // Check the number of parameters
    if (argc < 2) {
        // Tell the user how to run the program
        std::cerr << "Usage: " << argv[0] << "K" << std::endl;
        std::cerr << "Use K = 0 for regular mean" << std::endl;
        std::cerr << "For better mean, article recommends K = 25" << std::endl;
        return 1;
    }

    // Speed up stdio operations
    std::ios_base::sync_with_stdio(false);

    clock_t time0 = clock();
    SVDPlus* svd = new SVDPlus();
    clock_t time1 = clock();

    // Load in COO format into ratings vector
    svd->load("1.dta");
    /*
    double sum = 0.0;
    for (int i = 0; i < N_USERS; i++) {
        float Nu = (float) svd->rowIndex[i + 1] - svd->rowIndex[i];
        std::cout << "|N(u)| is " << Nu << std::endl;
        sum += (double) Nu;
    }
    std::cout << "Average |N(u)| is " << (sum / (double) N_USERS);
    */

    // Get baseline values.
    clock_t time2 = clock();
    Baseline* baseline = new Baseline();
    baseline->setK(atof(argv[1]));
    std::cout << "K = " << baseline->K <<std::endl;

    baseline->load("1.dta");

    std::cout << "Begin training\n";
    baseline->train("unused variable");
    float mu = baseline->global;
    float* user_bias = new float[N_USERS];
    float* movie_bias = new float[N_MOVIES];
    for (int i = 0; i < N_USERS; i++) {
        user_bias[i] = (float) baseline->average_array[i];
    }
    for (int i = 0; i < N_MOVIES; i++) {
        movie_bias[i] = (float) baseline->movie_average_array[i];
    }

    clock_t time3 = clock();

    std::cout << "Setting parameters" << std::endl;
    clock_t time4 = clock();
    svd->setParams(40, 0.001, 0.01, mu, user_bias, movie_bias);
    clock_t time5 = clock();

    std::cout << "Begin training" << std::endl;
    clock_t time6 = clock();
    svd->train("");
    clock_t time7 = clock();

    std::cout << "Printing output" << std::endl;
    svd->save("model/svd_plus/svd_plus.save");
    svd->printOutput("out/svd_plus.dta");
    clock_t time8 = clock();

    double ms1 = diffclock(time1, time0);
    std::cout << "Initializing took " << ms1 << " ms" << std::endl;
    double ms2 = diffclock(time2, time1);
    std::cout << "Total loading took " << ms2 << " ms" << std::endl;
    double ms2pt5 = diffclock(time3, time2);
    std::cout << "Baseline took " << ms2pt5 << " ms" << std::endl;
    double ms3 = diffclock(time5, time4);
    std::cout << "Setting params took " << ms3 << std::endl;;
    double ms4 = diffclock(time7, time6);
    std::cout << "Training took " << ms4 << std::endl;
    double ms5 = diffclock(time8, time7);
    std::cout << "Printing took " << ms5 << std::endl;
    double total_ms = diffclock(time8, time0);
    std::cout << "Total running time was " << total_ms << " ms" << std::endl;

    delete user_bias;
    delete movie_bias;
    return 0;
}
