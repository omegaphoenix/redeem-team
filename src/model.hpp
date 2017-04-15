#ifndef MODEL_HPP
#define MODEL_HPP
#include <string>
#include <time.h>
#include <vector>

#define N_MOVIES 17770
#define N_USERS 458293
#define N_DAYS 2243
#define N_TRAINING 99666409

class Model {
    public:
        Model();
        // std::vector<std::vector<float> > ratings;
        virtual ~Model();
        void loadFresh(std::string fname);
        void loadCSR(std::string fname);
        void loadSaved(std::string fname);
        virtual void train() = 0;
        void outputRatingsCSR(std::string fname);
        void outputRatingsRLE(std::string fname);
    private:
        virtual void generateMissing();
        std::vector<char> values;
        std::vector<int> columns;
        std::vector<int> rowIndex;
};

#endif // MODEL_HPP
