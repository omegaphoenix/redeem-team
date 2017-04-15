#ifndef MODEL_HPP
#define MODEL_HPP
#include <string>
#include <time.h>
#include <vector>

#define N_MOVIES 17770
#define N_USERS 458293
#define N_TRAINING 99666409

class Model {
    public:
        Model();
        // std::vector<std::vector<float> > ratings;
        virtual ~Model();
        void loadFresh(std::string fname);
        void loadSaved(std::string fname);
        virtual void train() = 0;
        void outputRatings(std::string fname);
    private:
        virtual void generateMissing();
        std::vector<char> values;
        std::vector<char> columns;
        std::vector<char> rowIndex;
};

#endif // MODEL_HPP
