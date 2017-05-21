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
    string trainFile = "data/um/1.dta";  //set train data
    string crossFile = "data/um/2.dta";  //set cross validation data
    string testFile = "data/um/4.dta";  //set test data
    string outFile = "out/test.txt";  //set output data
    TimeSVDPP svd(NULL,NULL,0,NULL,NULL, trainFile, crossFile, testFile, outFile);
    svd.train("");
    return 0;
}