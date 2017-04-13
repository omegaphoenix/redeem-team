#include <iostream>
#include <fstream>
#include "model.hpp"

// Clean up ratings array.
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
    data >> user >> movie >> date >> rating;
    std::cout << user << ' ' << movie << ' ' <<
              date << ' ' << rating << std::endl;
    data.close();
}

// Add in missing values.
void Model::generateMissing() {
}


// Load ratings array for a model in progress.
void Model::loadSaved(std::string fname) {
}
