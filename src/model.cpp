#include "model.hpp"
#include <fstream>
#include <iostream>

// Initialize ratings.
Model::Model() {
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
        std::vector<int> data_point(4, 0);
        data_point[0] = user;
        data_point[1] = movie;
        data_point[2] = date;
        data_point[3] = rating;
        ratings.push_back(data_point);
    }
    data.close();
}

// Add in missing values.
void Model::generateMissing() {
}


// Load ratings array for a model in progress.
void Model::loadSaved(std::string fname) {
}
