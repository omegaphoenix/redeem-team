#include "model.hpp"
#include <assert.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

// Initialize ratings.
Model::Model() {
}

// Clean up ratings.
Model::~Model() {
}

// Load new ratings array into CSR format.
void Model::loadFresh(std::string fname) {
    std::cout << "Opening " << fname << std::endl;

    FILE *f = std::fopen(fname.c_str(), "r");
    if (f == NULL) {
        throw std::runtime_error("Failed to open " + fname);
    }

    int user, movie, date, rating;
    int prevUser = 0;
    int itemsRead = fscanf(f,"%d %d %d %d\n", &user, &movie, &date, &rating);
    if (itemsRead != 4) {
        printf("Wrong format \n");
        return;
    }

    while (itemsRead  == 4) {
        assert (user > 0 && user <= N_USERS);
        assert (movie > 0 && movie <= N_MOVIES);
        assert (date > 0 && date <= N_DAYS);
        assert (rating >= 0 && rating <= 5);
        while (prevUser != user) {
            rowIndex.push_back(columns.size());
            prevUser++;
        }
        if (rating != 0) {
            values.push_back(rating);
            columns.push_back(movie - 1);
        }
        itemsRead = fscanf(f,"%d %d %d %d\n", &user, &movie, &date, &rating);
    }

    rowIndex.push_back(columns.size());
}

// Load ratings array from CSR format.
void Model::loadCSR(std::string fname) {
    // TODO
}

// Load ratings array from RLE format to COO.
void Model::loadRLE(std::string fname) {
    int f = open(fname.c_str(), O_RDONLY);
    unsigned short numZeroes;
    char rating;
    int user = 0;
    int movie = 0;

    off_t size = lseek(f, 0, SEEK_END);
    char *buffer = (char *) mmap(NULL, size, PROT_READ, MAP_PRIVATE, f, 0);

    int bytes = size;
    char *p = buffer;
    while (bytes > 0) {
        // Reached end of line/user
        if (*p == '\n') {
            p++;
            bytes--;
            user++;
            movie = 0;
        }
        // We have number of zeroes and a rating
        else {
            numZeroes = *p;
            p = p + sizeof(numZeroes);
            bytes = bytes - sizeof(numZeroes);
            movie += numZeroes;
            rating = *p;
            std::vector<int> data_point = {user, movie, 0, rating};
            p++;
            bytes--;
        }
    }
    close(f);
    munmap(buffer, size);
}

// Output integer ratings to file.
void Model::outputRatingsCSR(std::string fname) {
    int i;

    FILE *out = fopen((fname + "_CSR_values.dta").c_str(), "w");
    fprintf(out, "%lu\n", values.size());
    for (i = 0; i < values.size(); i++) {
        fprintf(out, "%c", values[i]);
    }
    fclose(out);

    out = fopen((fname + "_CSR_columns.dta").c_str(), "w");
    fprintf(out, "%lu\n", columns.size());
    for (i = 0; i < columns.size(); i++) {
        fprintf(out, "%d ", columns[i]);
    }
    fclose(out);

    out = fopen((fname + "_CSR_rowIndex.dta").c_str(), "w");
    fprintf(out, "%lu\n", rowIndex.size());
    for (i = 0; i < rowIndex.size(); i++) {
        fprintf(out, "%d ", rowIndex[i]);
    }
    fclose(out);
}

// Output integer ratings to file. Assume every user has at least one rating.
void Model::outputRatingsRLE(std::string fname) {
    int i;

    FILE *out = fopen((fname).c_str(), "w");
    // Index in values and columns vector
    int idx = 0;
    // Maximum repeated characters is N_MOVIES
    unsigned short numZeroes = 0;
    unsigned short colIdx = 0;

    for (i = 0; i + 1 < rowIndex.size(); i++) {
        // Index of next row/user in values and columns vectors
        int next = rowIndex[i + 1];
        if (i % 100000 == 0) {
            std::cout << i << std::endl;
        }

        // Output RLE sequence for i'th user
        while (idx < next) {
            // Check for zeroes in between
            assert (columns[idx] - colIdx < USHRT_MAX);
            numZeroes = columns[idx] - colIdx;
            fprintf(out, "%hu", numZeroes);

            // Next column to expect
            fprintf(out, "%c", values[idx]);
            colIdx = columns[idx] + 1;
            idx++;
        }

        colIdx = 0;
        // End line
        fprintf(out, "\n");
    }
    fclose(out);
}

// Add in missing values.
void Model::generateMissing() {
}


// Load ratings array for a model in progress.
void Model::loadSaved(std::string fname) {
}

// Run this function once first to preprocess data.
void Model::initLoad(std::string fname) {
    std::cout << "Preloading..." << std::endl;
    clock_t time0 = clock();

    // Load data from file.
    loadFresh("data/um/all.dta");
    clock_t time1 = clock();

    // Output ratings in new format.
    std::cout << "Outputing ratings" << std::endl;
    outputRatingsRLE(fname);
    clock_t time2 = clock();

    // Output times.
    double ms1 = diffclock(time1, time0);
    std::cout << "Preloading took " << ms1 << " ms" << std::endl;
    double ms2 = diffclock(time2, time1);
    std::cout << "Outputing data took " << ms2 << " ms" << std::endl;
    double total_ms = diffclock(time2, time0);
    std::cout << "Total preprocessing took " << total_ms << " ms" << std::endl;
}

// Load the data.
void Model::load(void) {
    std::string fname = "data/um/test_RLE.data";
    std::ifstream f(fname.c_str());
    if (!f.good()) {
        initLoad(fname);
    }

    std::cout << "Loading..." << std::endl;
    clock_t time0 = clock();
    // Load data from file.
    loadRLE(fname);
    clock_t time1 = clock();

    // Output times.
    double ms1 = diffclock(time1, time0);
    std::cout << "Loading took " << ms1 << " ms" << std::endl;
}

