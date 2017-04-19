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
        float K;

    private:
        void betterMean();
        // Only used for better mean
        float globalAverage();

        // For regular mean, k = 0
        // For better mean, k = integer
        // float K = 0;
        float *average_array;
        float *ratings_count;
};

#endif
