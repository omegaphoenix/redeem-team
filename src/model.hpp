#ifndef MODEL_HPP
#define MODEL_HPP
#include <array>
#include <string>
#include <time.h>
#include <vector>
#include "utils.hpp"

struct dataPoint {
    int userID;
    int movieID;
    int date;
    int value;

    dataPoint() {
        userID = 0;
        movieID = 0;
        date = 0;
        value = 0;
    }

    dataPoint(int a, int b, int c, int d) : userID(a), movieID(b), date(c), value(d) {
    }

    // Sort my movie
    bool operator<(const struct dataPoint &other) const
    {
        if (movieID != other.movieID) {
            return movieID < other.movieID;
        }
        else {
            return userID < other.userID;
        }
    }
};

class Model {
    public:
        Model();
        int* ratings; // COO format, movies and users 0-indexed
        unsigned int numRatings;
        unsigned char* values; // CSR values/ratings
        unsigned short* columns; // CSR columns/movies, 0-indexed
        unsigned int* rowIndex; // CSR row index, where user i starts in values/columns, 0-indexed

        // MU variables
        int* muratings; // COO format
        unsigned char* muvalues; // CSR values/ratings
        int* mucolumns; // CSR columns/user
        int* murowIndex; // CSR row index, where user i starts in values/columns
        dataPoint* sortStruct;     

        ~Model();
        void loadFresh(std::string inFname, std::string outFname);
        void loadCSR(std::string fname);
        // virtual void loadSaved(std::string fname) = 0;
        // virtual void train(std::string saveFile) = 0;
        void initLoad(std::string fname, std::string dataFile);
        void load(std::string dataFile);

        void transposeMU();
};

#endif // MODEL_HPP
