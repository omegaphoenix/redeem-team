#include "utils.hpp"
#include <assert.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <time.h>

int main(int argc, char **argv) {
    // Speed up stdio operations
    std::ios_base::sync_with_stdio(false);

    // Get filenames and check that they aren't the same
    std::string inFname = "out/timesvdpp/v0_timeSVDpp_0.89615RMSE"; // leave off the .txt
    std::string outFname = inFname + "_noisy.txt";
    inFname = inFname + ".txt";

    // Open files
    FILE* in = fopen(inFname.c_str(), "r");
    std::ofstream outputFile;
    outputFile << std::setprecision(3);
    outputFile.open(outFname);

    // Stream and output while adding noise
    float rating;
    int itemsRead = fscanf(in, "%f\n", &rating);
    bool noiseFlag = true;
    float noise = 0.25;
    int count = 0;
    while (itemsRead == 1) {
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
        outputFile << rating << "\n";
        itemsRead = fscanf(in, "%f\n", &rating);
    }

    // Close files
    fclose(in);
    outputFile.close();
    printf("Points modified: %d\n", count);
    return 0;
}
