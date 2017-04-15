#include "model.hpp"
#include <fstream>
#include <iostream>

// Initialize ratings.
Model::Model() : ratings(N_USERS, std::vector<float>(N_MOVIES, 0)) {
}

// Clean up ratings.
Model::~Model() {
}

// Load new ratings array.
void Model::loadFresh(std::string fname) {
    std::cout << "Opening " << fname << std::endl;
    std::ifstream data(fname);
    if (!data.is_open()) {
        throw std::runtime_error("Failed to open " + fname);
    }

    int user, movie, date;
    float rating;
    while (data.good()) {
      data >> user >> movie >> date >> rating;
      // Data is one indexed for users and movies
      ratings[user - 1][movie - 1] = rating;
    }
    data.close();
}

// Output integer ratings to file.
void Model::outputRatings(std::string fname) {
		std::ofstream out(fname);
    int i, j;

    for (i = 0; i < N_USERS; i++) {
        for (j = 0; j < N_MOVIES; j++) {
            char str[sizeof(int)];
            sprintf(str, "%d", (int) ratings[i][j]);
            out << str;
        }
        out << '\n';
    }
}

// Add in missing values.
void Model::generateMissing() {
}


// Load ratings array for a model in progress.
void Model::loadSaved(std::string fname) {
}
