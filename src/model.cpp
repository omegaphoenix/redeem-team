#include "model.hpp"
#include <assert.h>
#include <climits>
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
    values = new unsigned char[N_TRAINING];
    columns = new unsigned short[N_TRAINING];
    rowIndex = new unsigned int[N_USERS + 1];
}

// Clean up ratings.
Model::~Model() {
    delete ratings;
    delete values;
    delete columns;
    delete rowIndex;
}

// Load new ratings array into CSR format.
void Model::loadFresh(std::string inFname, std::string outFname) {
    printf("Opening %s\n", inFname.c_str());

    FILE* in = fopen(inFname.c_str(), "r");
    FILE* out = fopen((outFname).c_str(), "wb");
    unsigned char high, low, val;
    unsigned short movieNo;
    unsigned char newuser = UCHAR_MAX;
    if (in == NULL) {
        throw std::runtime_error("Failed to open " + inFname);
    }

    int user, movie, date, rating;
    int prevUser = 1;
    int itemsRead = fscanf(in,"%d %d %d %d\n", &user, &movie, &date, &rating);
    if (itemsRead != DATA_POINT_SIZE) {
        throw std::runtime_error("Wrong format");
    }

    while (itemsRead  == DATA_POINT_SIZE) {
        assert (user > 0 && user <= N_USERS);
        assert (movie > 0 && movie <= N_MOVIES);
        assert (date > 0 && date <= N_DAYS);
        assert (rating >= 0 && rating <= MAX_RATING);
        while (prevUser != user) {
            prevUser++;
            fwrite(&newuser, sizeof(unsigned char), 1, out);
            fwrite(&newuser, sizeof(unsigned char), 1, out);
        }
        if (rating != 0) {
            movie--;
            movieNo = (unsigned short) movie;
            assert (movieNo >= 0 && movieNo < N_MOVIES);
            high = (movieNo >> CHAR_BIT) & UCHAR_MAX;
            low = movieNo & UCHAR_MAX;
            fwrite(&high, sizeof(unsigned char), 1, out);
            fwrite(&low, sizeof(unsigned char), 1, out);
            val = (unsigned char) rating;
            assert (val > 0 && val <= MAX_RATING);
            fwrite(&val, sizeof(unsigned char), 1, out);
        }
        itemsRead = fscanf(in,"%d %d %d %d\n", &user, &movie, &date, &rating);
    }
    fwrite(&newuser, sizeof(unsigned char), 1, out);
    fwrite(&newuser, sizeof(unsigned char), 1, out);
    fclose(in);
    fclose(out);
}

// Load ratings array from CSR format to COO and CSR.
void Model::loadCSR(std::string fname) {
    int f = open(fname.c_str(), O_RDONLY);
    unsigned char rating, high, low;
    int user = 0;
    unsigned short movie = 0;

    off_t size = lseek(f, 0, SEEK_END);
    unsigned char* buffer = (unsigned char*) mmap(NULL, size, PROT_READ, MAP_PRIVATE, f, 0);

    int bytes = size;
    int ratingsIdx = 0;
    int idx = 0;
    rowIndex[user] = idx;
    // short for end of user marker, short + char per data point
    int numPoints = (bytes - N_USERS * sizeof(short))
                    / (sizeof(short) + sizeof(char));
    assert (numPoints <= N_TRAINING);
    unsigned char* p = buffer;
    while (bytes > 0) {
        // Reached end of line/user if two bytes are 0xff and 0xff
        // This was chosen because no movie can go up to USHRT_MAX
        if (*p == UCHAR_MAX && *(p + 1) == UCHAR_MAX) {
            // Move 2 bytes for end of user marker
            p += sizeof(short);
            bytes -= sizeof(short);
            user++;
            assert(user <= N_USERS);
            rowIndex[user] = idx;
        }
        // We have number of zeroes and a rating
        else {
            high = *p;
            p++;
            bytes--;
            movie = high;
            movie = movie << CHAR_BIT;
            low = *p;
            p++;
            bytes--;
            movie += low;
            rating = *p;
            p++;
            bytes--;
            assert (movie >= 0 && movie < N_MOVIES);
            assert (rating >= 0 && rating <= MAX_RATING);
            ratings[ratingsIdx + USER_IDX] = user;
            ratings[ratingsIdx + MOVIE_IDX] = movie;
            ratings[ratingsIdx + RATING_IDX] = rating;
            values[idx] = rating;
            columns[idx] = movie;
            idx++;
            ratingsIdx += DATA_POINT_SIZE;
        }
    }
    numRatings = idx;
    assert (numRatings == rowIndex[user]);
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
void Model::initLoad(std::string fname, std::string dataFile) {
    debugPrint("Preprocessing...\n");
    clock_t time0 = clock();

    // Load data from file.
    loadFresh("data/um/" + dataFile, fname);
    clock_t time1 = clock();

    // Output times.
    double ms1 = diffclock(time1, time0);
    printf("Preproccessing took %f ms\n", ms1);
}

// Load the data.
void Model::load(std::string dataFile) {
    std::string fname = "data/um/" + dataFile + ".csr";
    std::ifstream f(fname.c_str());
    if (!f.good()) {
        initLoad(fname, dataFile);
    }

    debugPrint("Loading...\n");
    clock_t time0 = clock();
    // Load data from file.
    loadCSR(fname);
    clock_t time1 = clock();

    // Output times.
    double ms1 = diffclock(time1, time0);
    printf("Loading took %f ms\n", ms1);
}
