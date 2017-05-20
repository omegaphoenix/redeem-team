int main(int argc, char **argv) {
    // Check the number of parameters
    if (argc < 2) {
        // Tell the user how to run the program
        std::cerr << "Usage: " << argv[0] << "K" << std::endl;
        std::cerr << "Use K = 0 for regular mean" << std::endl;
        std::cerr << "For better mean, article recommends K = 25" << std::endl;
        return 1;
    }

    clock_t time0 = clock();
    Baseline* baseline = new Baseline();
    baseline->setK(atof(argv[1]));
    std::cout << "K = " << baseline->K <<std::endl;

    clock_t time1 = clock();

    // Load data from file.
    baseline->load("1.dta");
    clock_t time2 = clock();

    // Train by building correlation matrix
    std::cout << "Begin training\n";
    baseline->train("unused variable");
    /*
    for(int i = 0; i < N_USERS; ++i) {
        std::cout << "avg " << baseline->average_array[i] << "\n";
        std::cout << "stdev " << baseline->stdev_array[i] << "\n\n";
        if (isnan(baseline->average_array[i]) || isnan(baseline->stdev_array[i])) {
            std::cout << "NaN" << std::endl;
        }
    }*/

    clock_t time3 = clock();

    // Output times.
    double ms1 = diffclock(time1, time0);
    std::cout << "Initialization took " << ms1 << " ms" << std::endl;
    double ms2 = diffclock(time2, time1);
    std::cout << "Loading took " << ms2 << " ms" << std::endl;
    double total_ms = diffclock(time2, time0);
    std::cout << "Total took " << total_ms << " ms" << std::endl;

    double baseline_ms = diffclock(time3, time0);
    std::cout << "baseline took " << baseline_ms << " ms" << std::endl;

    return 0;
}