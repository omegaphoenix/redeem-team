#include <iostream>
#include <time.h>
#include "naive_svd.hpp"

int main(int argc, char **argv) {
    // Speed up stdio operations
    std::ios_base::sync_with_stdio(false);

    clock_t time0 = clock();
    NaiveSVD* nsvd = new NaiveSVD();
    clock_t time1 = clock();

    // Load in COO format into ratings vector
    nsvd->load("1.dta");
    clock_t time2 = clock();

    std::cout << "Setting parameters" << std::endl;
    clock_t time3 = clock();
    nsvd->setParams(40, 0.001, 0.0);
    clock_t time4 = clock();

    std::cout << "Begin training" << std::endl;
    clock_t time5 = clock();
    nsvd->train("model/naive_svd/naive_svd.save");
    clock_t time6 = clock();

    std::cout << "Printing output" << std::endl;
    nsvd->save("model/naive_svd/naive_svd.save");
    nsvd->printOutput("out/naive_svd.dta");
    clock_t time7 = clock();

    double ms1 = diffclock(time1, time0);
    std::cout << "Initializing took " << ms1 << " ms" << std::endl;
    double ms2 = diffclock(time2, time1);
    std::cout << "Total loading took " << ms2 << " ms" << std::endl;
    double ms3 = diffclock(time4, time3);
    std::cout << "Setting params took " << ms3 << std::endl;;
    double ms4 = diffclock(time6, time5);
    std::cout << "Training took " << ms4 << std::endl;
    double ms5 = diffclock(time7, time6);
    std::cout << "Printing took " << ms5 << std::endl;
    double total_ms = diffclock(time7, time0);
    std::cout << "Total running time was " << total_ms << " ms" << std::endl;
    return 0;
}
