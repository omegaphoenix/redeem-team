//This file is used to generate predictions.
#include <cmath>
#include <iostream>
#include <cstring> 
#include <cstdlib>
#include <fstream>
#include <ctime>
#include <string>
#include <string.h> 
#include <utility>
#include "time_svd_pp.hpp"

using namespace std;

int main() {
    string trainFile = "1.dta";  //set cross validation data
    string crossFile = "data/um/4.dta";  //set cross validation data
    string testFile = "data/um/5-1.dta";  //set test data

    // Run this after saving the 2 epoch run
    /*
    TimeSVDPP* load = loadTSVDpp("model/timesvdpp/50factors_30bins_2epochs.save",
        trainFile, crossFile, testFile);
    load->train("");
    */

    TimeSVDPP svd(false,0,0,0,NULL,NULL,NULL,NULL,
                  NULL,NULL,NULL,NULL,NULL,NULL,
                  trainFile, crossFile, testFile);
    svd.train("");
    return 0;
}
