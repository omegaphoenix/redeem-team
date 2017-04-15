#include "model.hpp"
#include <fstream>
#include <iostream>

// Initialize ratings.
Model::Model() { // : ratings(N_USERS, std::vector<float>(N_MOVIES, 0)) {
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
    int prevUser = 0;
    float rating;
    while (data.good()) {
        data >> user >> movie >> date >> rating;
        // Data is one indexed for users and movies
        // ratings[user - 1][movie - 1] = rating;
        while (prevUser != user) {
            rowIndex.push_back(columns.size());
            prevUser++;
        }
        values.push_back(rating);
        columns.push_back(movie);
    }
    rowIndex.push_back(columns.size());
    data.close();
}

// Output integer ratings to file.
void Model::outputRatings(std::string fname) {
		std::ofstream out(fname);
    int i;

    for (i = 0; i < values.size(); i++) {
        if (i % 10000000 == 0) {
            std::cout << i << std::endl;
        }
        char str[sizeof(char)];
        sprintf(str, "%c", (char) values[i]);
        out << str;
    }
    out << '\n';

    for (i = 0; i < columns.size(); i++) {
        char str[sizeof(char)];
        sprintf(str, "%c ", (char) columns[i]);
        out << str;
    }
    out << '\n';

    for (i = 0; i < rowIndex.size(); i++) {
        char str[sizeof(char)];
        sprintf(str, "%c ", (char) rowIndex[i]);
        out << str;
    }
    out << '\n';
}

// Add in missing values.
void Model::generateMissing() {
}


// Load ratings array for a model in progress.
void Model::loadSaved(std::string fname) {
}
