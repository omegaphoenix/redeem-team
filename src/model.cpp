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
    ratings = new int[N_TRAINING * DATA_POINT_SIZE];
}

// Clean up ratings.
Model::~Model() {
    free(ratings);
}

// Load new ratings array into CSR format.
void Model::loadFresh(std::string inFname, std::string outFname) {
    std::cout << "Opening " << inFname << std::endl;

    FILE* in = fopen(inFname.c_str(), "r");
    FILE* out = fopen((outFname).c_str(), "wb");
    unsigned char high, low, val;
    unsigned short movieNo;
    unsigned char newuser = 0xff;
    if (in == NULL) {
        throw std::runtime_error("Failed to open " + inFname);
    }

    int user, movie, date, rating;
    int prevUser = 1;
    int itemsRead = fscanf(in,"%d %d %d %d\n", &user, &movie, &date, &rating);
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
            prevUser++;
            fwrite(&newuser, sizeof(unsigned char), 1, out);
            fwrite(&newuser, sizeof(unsigned char), 1, out);
        }
        if (rating != 0) {
            movie--;
            movieNo = (unsigned short) movie;
            assert (movieNo >= 0 && movieNo < N_MOVIES);
            high = (movieNo >> 8) & 0xff;
            low = movieNo & 0xff;
            fwrite(&high, sizeof(unsigned char), 1, out);
            fwrite(&low, sizeof(unsigned char), 1, out);
            val = (unsigned char) rating;
            assert (val > 0 && val <= 5);
            fwrite(&val, sizeof(unsigned char), 1, out);
        }
        itemsRead = fscanf(in,"%d %d %d %d\n", &user, &movie, &date, &rating);
    }
    fwrite(&newuser, sizeof(unsigned char), 1, out);
    fwrite(&newuser, sizeof(unsigned char), 1, out);
    fclose(in);
    fclose(out);
}

// Output integer ratings to file. Assume every user has at least one rating.
void Model::outputRatingsCSR(std::string fname) {
    int i;

    FILE* out = fopen((fname).c_str(), "wb");
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

        // Output CSR sequence for i'th user
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

// Load ratings array from CSR format to COO.
void Model::loadCSR(std::string fname) {
    int f = open(fname.c_str(), O_RDONLY);
    unsigned char rating, high, low;
    int user = 0;
    unsigned short movie = 0;

    off_t size = lseek(f, 0, SEEK_END);
    unsigned char* buffer = (unsigned char*) mmap(NULL, size, PROT_READ, MAP_PRIVATE, f, 0);

    int bytes = size;
    int idx = 0;
    int numPoints = (bytes - N_USERS * 2) / 3;
    assert (numPoints <= N_TRAINING);
    unsigned char* p = buffer;
    while (bytes > 0) {
        // Reached end of line/user
        if (*p == 0xff && *(p + 1) == 0xff) {
            // Move 2 bytes for end of user marker
            p += sizeof(short);
            bytes -= sizeof(short);
            user++;
            assert(user <= N_USERS);
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
            ratings[idx + USER_IDX] = user;
            ratings[idx + MOVIE_IDX] = movie;
            ratings[idx + RATING_IDX] = rating;
            idx += DATA_POINT_SIZE;
        }
    }
    close(f);
    munmap(buffer, size);
}

// Add in missing values.
void Model::generateMissing(void) {
}


// Load ratings array for a model in progress.
void Model::loadSaved(std::string fname) {
}

// Run this function once first to preprocess data.
void Model::initLoad(std::string fname) {
    std::cout << "Preprocessing..." << std::endl;
    clock_t time0 = clock();

    // Load data from file.
    loadFresh("data/um/all.dta", fname);
    clock_t time1 = clock();

    // Output times.
    double ms1 = diffclock(time1, time0);
    std::cout << "Preprocessing took " << ms1 << " ms" << std::endl;
}

// Load the data.
void Model::load(void) {
    std::string fname = "data/um/test_CSR.dta";
    std::ifstream f(fname.c_str());
    if (!f.good()) {
        initLoad(fname);
    }

    std::cout << "Loading..." << std::endl;
    clock_t time0 = clock();
    // Load data from file.
    loadCSR(fname);
    clock_t time1 = clock();

    // Output times.
    double ms1 = diffclock(time1, time0);
    std::cout << "Loading took " << ms1 << " ms" << std::endl;
}

