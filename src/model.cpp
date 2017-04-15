#include "model.hpp"
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
Model::Model() { // : ratings(N_USERS, std::vector<float>(N_MOVIES, 0)) {
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
        while (prevUser != user) {
            rowIndex.push_back(columns.size());
            prevUser++;
        }
        values.push_back(rating);
        columns.push_back(movie);
        itemsRead = fscanf(f,"%d %d %d %d\n", &user, &movie, &date, &rating);
    }
}

// Load ratings array in CSR format
void Model::loadCSR(std::string fname) {
    // TODO
}

// Output integer ratings to file.
void Model::outputRatingsCSR(std::string fname) {
    FILE *out = fopen(fname.c_str(), "w");
    int i;

    fprintf(out, "%lu\n", values.size());
    fprintf(out, "%lu\n", columns.size());
    fprintf(out, "%lu\n", rowIndex.size());

    for (i = 0; i < values.size(); i++) {
        if (i % 10000000 == 0) {
            std::cout << i << std::endl;
        }
        fprintf(out, "%c", values[i]);
    }
    fprintf(out, "\n");

    for (i = 0; i < columns.size(); i++) {
        fprintf(out, "%d ", columns[i]);
    }
    fprintf(out, "\n");

    for (i = 0; i < rowIndex.size(); i++) {
        fprintf(out, "%d ", rowIndex[i]);
    }
    fprintf(out, "\n");
    fclose(out);
}

// Output integer ratings to file.
void Model::outputRatingsRLE(std::string fname) {
    // TODO
}

// Add in missing values.
void Model::generateMissing() {
}


// Load ratings array for a model in progress.
void Model::loadSaved(std::string fname) {
}
