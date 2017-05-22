#include "model.hpp"
#include <assert.h>
#include <climits>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>

// Initialize ratings.
Model::Model() {
    // UM
#if defined(COO) || defined(MU)
    ratings = new int[N_TRAINING * DATA_POINT_SIZE];
#endif
    values = new unsigned char[N_TRAINING];
    columns = new unsigned short[N_TRAINING];
    dates = new unsigned short[N_TRAINING];
    rowIndex = new unsigned int[N_USERS + 1];

    // MU
#ifdef MU
    sortStruct = new dataPoint[N_TRAINING];
    muratings = new int[N_TRAINING * DATA_POINT_SIZE];
    muvalues = new unsigned char[N_TRAINING];
    mucolumns = new int[N_TRAINING];
    mudates = new int[N_TRAINING];
    murowIndex = new int[N_MOVIES + 1];
#endif
}

// Clean up ratings.
Model::~Model() {
#if defined(COO) || defined(MU)
    delete ratings;
#endif
    delete values;
    delete columns;
    delete dates;
    delete rowIndex;

#ifdef MU
    delete sortStruct;
    delete muratings;
    delete muvalues;
    delete mucolumns;
    delete mudates;
    delete murowIndex;
#endif
}

#ifdef MU
void Model::transposeMU() {
    clock_t time0 = clock();
    for (unsigned int i = 0; i < numRatings; i++) {
        int u = ratings[i * DATA_POINT_SIZE + USER_IDX];
        int m = ratings[i * DATA_POINT_SIZE + MOVIE_IDX];
        int d = ratings[i * DATA_POINT_SIZE + TIME_IDX];
        int r = ratings[i * DATA_POINT_SIZE + RATING_IDX];
        sortStruct[i] = dataPoint(u, m, d, r);
    }
    std::sort(sortStruct, sortStruct + numRatings);

    int current = 0; //CSR counter

    for (unsigned int i = 0; i < numRatings; i++) {
        // COO Format
        muratings[i * DATA_POINT_SIZE + USER_IDX] = sortStruct[i].userID;
        muratings[i * DATA_POINT_SIZE + MOVIE_IDX] = sortStruct[i].movieID;
        muratings[i * DATA_POINT_SIZE + TIME_IDX] = sortStruct[i].date;
        muratings[i * DATA_POINT_SIZE + RATING_IDX] = sortStruct[i].value;

        // CSR Format
        while (sortStruct[i].movieID > current) {
            current++;
            murowIndex[current] = i;
        }
        muvalues[i] = sortStruct[i].value;
        mucolumns[i] = sortStruct[i].userID;
        mudates[i] = sortStruct[i].date;
    }
    murowIndex[N_MOVIES] = numRatings;
    clock_t time1 = clock();

    float ms1 = diffclock(time1, time0);
    printf("Transposing took %f ms\n", ms1);;
}
#endif

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
            date--;
            movieNo = (unsigned short) movie;
            assert (movieNo >= 0 && movieNo < N_MOVIES);
            high = (movieNo >> CHAR_BIT) & UCHAR_MAX;
            low = movieNo & UCHAR_MAX;
            fwrite(&high, sizeof(unsigned char), 1, out);
            fwrite(&low, sizeof(unsigned char), 1, out);
            assert (date >= 0 && date < N_DAYS);
            high = (date >> CHAR_BIT) & UCHAR_MAX;
            low = date & UCHAR_MAX;
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
    unsigned short date = 0;

    off_t size = lseek(f, 0, SEEK_END);
    unsigned char* buffer = (unsigned char*) mmap(NULL, size, PROT_READ, MAP_PRIVATE, f, 0);

    int bytes = size;
#if defined(COO) || defined(MU)
    int ratingsIdx = 0;
#endif
    int idx = 0;
    rowIndex[user] = idx;
    // short for end of user marker, short + short + char per data point
#ifndef NDEBUG
    int numPoints = (bytes - N_USERS * sizeof(short))
                    / (2 * sizeof(short) + sizeof(char));
    assert (numPoints <= N_TRAINING);
#endif
    unsigned char* p = buffer;
    while (bytes > 0) {
        // Reached end of line/user if two bytes are 0xff and 0xff
        // This was chosen because no movie can go up to USHRT_MAX
        if (*p == UCHAR_MAX && *(p + 1) == UCHAR_MAX) {
            // Move 2 bytes for end of user marker
            p += sizeof(short);
            bytes -= sizeof(short);
            user++;
            assert (user <= N_USERS);
            rowIndex[user] = idx;
        }
        // We have number of zeroes and a rating
        else {
            // Read movie
            high = *p;
            p++;
            bytes--;
            movie = high;
            movie = movie << CHAR_BIT;
            low = *p;
            p++;
            bytes--;
            movie += low;
            // Read date
            high = *p;
            p++;
            bytes--;
            date = high;
            date = date << CHAR_BIT;
            low = *p;
            p++;
            bytes--;
            date += low;
            // Read rating
            rating = *p;
            p++;
            bytes--;
            // Validate
            assert (movie >= 0 && movie < N_MOVIES);
            assert (date >= 0 && date < N_DAYS);
            assert (rating >= 0 && rating <= MAX_RATING);
            // Save data
#if defined(COO) || defined(MU)
            ratings[ratingsIdx + USER_IDX] = user;
            ratings[ratingsIdx + MOVIE_IDX] = movie;
            ratings[ratingsIdx + TIME_IDX] = date;
            ratings[ratingsIdx + RATING_IDX] = rating;
#endif
            values[idx] = rating;
            columns[idx] = movie;
            dates[idx] = date;
            idx++;
#if defined(COO) || defined(MU)
            ratingsIdx += DATA_POINT_SIZE;
#endif
        }
    }
    numRatings = idx;
    assert (numRatings == rowIndex[user]);
    close(f);
    munmap(buffer, size);
}

// Run this function once first to preprocess data.
void Model::initLoad(std::string fname, std::string dataFile) {
    debugPrint("Preprocessing...\n");
    clock_t time0 = clock();

    // Load data from file.
    loadFresh("data/um/" + dataFile, fname);
    clock_t time1 = clock();

    // Output times.
    float ms1 = diffclock(time1, time0);
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
    float ms1 = diffclock(time1, time0);
    printf("Loading took %f ms\n", ms1);
}

// Return RMSE on validation file
float Model::validate(std::string valFile) {
    debugPrint("Validating...\n");
    clock_t timeStart = clock();
    Model *validator = new Model();
    validator->load(valFile);
    unsigned int userStartIdx, userEndIdx, n, i, k, colIdx;
    float squareError = 0.0;
    for (n = 0; n < N_USERS; ++n) {
#ifdef ISRBM
        prepPredict(validator, n);
#endif
        userStartIdx = validator->rowIndex[n];
        userEndIdx = validator->rowIndex[n + 1];
        for (colIdx = userStartIdx; colIdx < userEndIdx; colIdx++) {
            i = validator->columns[colIdx]; // movie
            k = validator->values[colIdx]; // rating
            assert (i >= 0 && i < N_MOVIES);
            assert (k > 0 && k <= MAX_RATING);
            unsigned int d = validator->dates[colIdx]; // date
            float prediction = predict(n, i, d); // jump
            float error = prediction - (float) k;
            squareError += error * error;
            assert (squareError >= 0); // jump
        }
    }
    clock_t timeEnd = clock();
    float msTotal = diffclock(timeEnd, timeStart);
    printf("Validation took %f ms\n", msTotal);
    float RMSE = sqrt(squareError / validator->numRatings);
    delete validator;
    return RMSE;
}

// Return RMSE on validation file
float Model::trainingError() {
    debugPrint("Calculating training error...\n");
    clock_t timeStart = clock();
    unsigned int userStartIdx, userEndIdx, n, i, k, colIdx;
    float squareError = 0.0;
    for (n = 0; n < N_USERS; ++n) {
#ifdef ISRBM
        prepPredict(this, n);
#endif
        userStartIdx = rowIndex[n];
        userEndIdx = rowIndex[n + 1];
        for (colIdx = userStartIdx; colIdx < userEndIdx; colIdx++) {
            i = columns[colIdx]; // movie
            k = values[colIdx]; // rating
            assert (i >= 0 && i < N_MOVIES);
            assert (k > 0 && k <= MAX_RATING);
            unsigned int d = dates[colIdx]; // date
            float prediction = predict(n, i, d); // jump
            float error = prediction - (float) k;
            squareError += error * error;
            assert (squareError >= 0);
        }
    }
    clock_t timeEnd = clock();
    float msTotal = diffclock(timeEnd, timeStart);
    printf("Calculating training error took %f ms\n", msTotal);
    float RMSE = sqrt(squareError / numRatings);
    return RMSE;
}

// Output submission
void Model::output(std::string saveFile) {
    debugPrint("Outputing...\n");
    clock_t timeStart = clock();
    Model *validator = new Model();
    validator->load("5-1.dta");
    unsigned int userStartIdx, userEndIdx, n, i, colIdx;

    // Open file
    std::ofstream outputFile;
    outputFile << std::setprecision(4);
    outputFile.open(saveFile);
    for (n = 0; n < N_USERS; ++n) {
#ifdef ISRBM
        prepPredict(validator, n);
#endif
        userStartIdx = validator->rowIndex[n];
        userEndIdx = validator->rowIndex[n + 1];
        for (colIdx = userStartIdx; colIdx < userEndIdx;
                colIdx++) {
            i = validator->columns[colIdx]; // movie
            assert (i >= 0 && i < N_MOVIES);
            unsigned int d = validator->dates[colIdx]; // date
            float prediction = predict(n, i, d); // jump
            outputFile << prediction << "\n"; // jump
        }
    }
    outputFile.close();

    clock_t timeEnd = clock();
    float msTotal = diffclock(timeEnd, timeStart);
    printf("Outputing took %f ms\n", msTotal);

    delete validator;
}

void Model::train(std::string saveFile) {
    printf("Not doing anything on %s\n", saveFile.c_str());
}

#ifdef ISRBM
void Model::prepPredict(Model *mod, int n) {
}
#endif

float Model::predict(int n, int i, int d) {
    printf("Not predicting %d %d %d\n", n, i, d);
    return 0.0;
}

#ifdef MU
void testTranspose() {
#ifndef NDEBUG
    clock_t time0 = clock();
    Model* mod = new Model();
    clock_t time1 = clock();
    mod->load("1.dta");
    clock_t time2 = clock();
    printf("Transpose\n");
    mod->transposeMU();
    clock_t time3 = clock();
    unsigned int i;
    for (i = 0; i < mod->numRatings - 1; i++) {
        int j = i + 1;
        // COO values
        int user = mod->muratings[i * DATA_POINT_SIZE + USER_IDX];
        int nextUser = mod->muratings[j * DATA_POINT_SIZE + USER_IDX];
        int movie = mod->muratings[i * DATA_POINT_SIZE + MOVIE_IDX];
        int nextMovie = mod->muratings[j * DATA_POINT_SIZE + MOVIE_IDX];

        // CSR values
        int csrUser = mod->mucolumns[i];
        int csrRating = mod->muvalues[i];
        assert (user == csrUser);

        // Check order of COO
        assert (movie <= nextMovie);
        if (movie == nextMovie) {
            assert (user <= nextUser);
        }

        // Check order of CSR
        assert (csrRating > 0 && csrRating <= MAX_RATING);
        if (i < N_MOVIES) {
            int csrUserIdx = mod->murowIndex[i];
            int csrNextUserIdx = mod->murowIndex[j];
            assert (csrUserIdx <= csrNextUserIdx);
        }
    }

    assert (mod->murowIndex[0] == 0);
    assert (mod->murowIndex[N_MOVIES] == (int) mod->numRatings);
    clock_t time4 = clock();

    float ms1 = diffclock(time1, time0);
    float ms2 = diffclock(time2, time1);
    float ms3 = diffclock(time3, time2);
    float ms4 = diffclock(time4, time3);

    printf("Initializing took %f ms\n", ms1);
    printf("Total loading took %f ms\n", ms2);
    printf("Transposing took %f ms\n", ms3);
    printf("Testing took %f ms\n", ms4);
#endif
}
#endif
