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
        double* average_array;
        float* ratings_count;
        double* stdev_array;
        double* movie_average_array;
        float* movie_count;
        float global;

    private:
        void betterMean();
        void standardDeviation();
        void movieMean();
        // Only used for better mean
        void globalAverage();
};

#endif
