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

// Returns the differences in ms.
static double diffclock(clock_t clock1, clock_t clock2) {
  double diffticks = clock1 - clock2;
  double diffms = (diffticks) / (CLOCKS_PER_SEC / 1000);
  return diffms;
}

class Model {
    public:
        Model();
        std::vector<std::array<int, 4> > ratings;
        virtual ~Model();
        void loadFresh(std::string fname);
        void loadCSR(std::string fname);
        void loadRLE(std::string fname);
        void loadSaved(std::string fname);
        virtual void train() = 0;
        void outputRatingsCSR(std::string fname);
        void outputRatingsRLE(std::string fname);
        void initLoad(std::string fname);
        void load(void);
    private:
        virtual void generateMissing();
        std::vector<unsigned char> values;
        std::vector<unsigned short> columns;
        std::vector<int> rowIndex;
};

#endif // MODEL_HPP
