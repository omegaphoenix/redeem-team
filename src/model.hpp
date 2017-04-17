#ifndef MODEL_HPP
#define MODEL_HPP
#include <array>
#include <string>
#include <time.h>
#include <vector>

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

// Returns the differences in ms.
static double diffclock(clock_t clock1, clock_t clock2) {
    double diffticks = clock1 - clock2;
    double diffms = (diffticks) / (CLOCKS_PER_SEC / 1000);
    return diffms;
}

class Model {
    public:
        Model();
        int* ratings;
        int ratings_size;
        virtual ~Model();
        void loadFresh(std::string inFname, std::string outFname);
        void loadCSR(std::string fname);
        void loadSaved(std::string fname);
        virtual void train() = 0;
        void initLoad(std::string fname);
        void load(void);
    private:
        virtual void generateMissing(void);
        std::vector<unsigned char> values;
        std::vector<unsigned short> columns;
        std::vector<int> rowIndex;
};

#endif // MODEL_HPP
