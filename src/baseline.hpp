#ifndef BASELINE
#define BASELINE
#include "model.hpp"

class Baseline : public Model {
    public:
        Baseline();
        ~Baseline();
        void train(std::string saveFile);
        void loadSaved(std::string fname);
        void setK(float constant);
        // For regular mean, k = 0
        // For better mean, k = integer
        float K;
        float* average_array;
        float* ratings_count;
        float* stdev_array;
        float* movie_average_array;
        float* movie_count;

    private:
        void betterMean();
        void standardDeviation();
        void movieMean();
        // Only used for better mean
        float globalAverage();
};

#endif
