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
Model::Model() : ratings(N_TRAINING, std::vector<int>(4, 0)) {
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
            values.push_back((unsigned char) rating);
            columns.push_back((unsigned short) (movie - 1));
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
    unsigned char rating, high, low;
    int user = 0;
    unsigned short movie = 0;

    off_t size = lseek(f, 0, SEEK_END);
    unsigned char *buffer = (unsigned char *) mmap(NULL, size, PROT_READ, MAP_PRIVATE, f, 0);

    int bytes = size;
    int i = 0;
    int numPoints = (bytes - N_USERS * 2) / 3;
    assert (numPoints <= N_TRAINING);
    unsigned char *p = buffer;
    while (bytes > 0) {
        // Reached end of line/user
        if (*p == 0xff && *(p + 1) == 0xff) {
            // Move 2 bytes for end of user marker
            p += sizeof(short);
            bytes -= sizeof(short);
            user++;
        }
        // We have number of zeroes and a rating
        else {
            high = *p;
            p++;
            bytes--;
            movie = high;
            movie = movie << 8;
            low = *p;
            p++;
            bytes--;
            movie += low;
            rating = *p;
            p++;
            bytes--;
            assert (movie >= 0 && movie < N_MOVIES);
            assert (rating >= 0 && rating <= 5);
            ratings[i][0] = user;
            ratings[i][1] = movie;
            ratings[i][3] = rating;
            i++;
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
        fprintf(out, "%hu ", columns[i]);
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

    FILE *out = fopen((fname).c_str(), "wb");
    // Index in values and columns vector
    int idx = 0;
    unsigned char high, low;
    unsigned char newuser = 0xff;

    for (i = 0; i + 1 < rowIndex.size(); i++) {
        // Index of next row/user in values and columns vectors
        int next = rowIndex[i + 1];
        if (i % 100000 == 0) {
            std::cout << i << std::endl;
        }

        // Output RLE sequence for i'th user
        while (idx < next) {
            // Check for zeroes in between
            assert (columns[idx] >= 0 && columns[idx] < N_MOVIES);
            high = (columns[idx] >> 8) & 0xff;
            low = columns[idx] & 0xff;
            // fwrite(&(columns[idx]), sizeof(unsigned short), 1, out);
            fwrite(&high, sizeof(unsigned char), 1, out);
            fwrite(&low, sizeof(unsigned char), 1, out);
            assert (values[idx] >= 0 && values[idx] <= 5);
            fwrite(&(values[idx]), sizeof(unsigned char), 1, out);
            idx++;
        }

        // Two 0xff's indicate a new user
        // This is greater than N_MOVIES
        fwrite(&newuser, sizeof(unsigned char), 1, out);
        fwrite(&newuser, sizeof(unsigned char), 1, out);
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
    std::string fname = "data/um/test_RLE.dta";
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

