#ifndef MODEL_HPP
#define MODEL_HPP
#include <array>
#include <string>
#include <time.h>
#include <vector>
#include "utils.hpp"

// Disable assertions
// #define NDEBUG

#define N_MOVIES 17770
#define N_USERS 458293
#define N_DAYS 2243
#define N_TRAINING 99666408
#define MAX_RATING 5
#define USER_IDX 0
#define MOVIE_IDX 1
#define TIME_IDX 2
#define RATING_IDX 3
#define DATA_POINT_SIZE 4

class Model {
    public:
        Model();
        int* ratings; // COO format
        int numRatings;
        unsigned char* values; // CSR values/ratings
        unsigned short* columns; // CSR columns/movies
        int* rowIndex; // CSR row index, where user i starts in values/columns
        virtual ~Model();
        void loadFresh(std::string inFname, std::string outFname);
        void loadCSR(std::string fname);
        virtual void loadSaved(std::string fname) = 0;
        virtual void train(std::string saveFile) = 0;
        void initLoad(std::string fname, std::string dataFile);
        void load(std::string dataFile);
    private:
        virtual void generateMissing(void);
};

#endif // MODEL_HPP
