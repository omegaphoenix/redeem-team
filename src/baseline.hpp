#ifndef BASELINE
#define BASELINE
#include "model.hpp"

class Baseline : public Model {
    public:
        Baseline();
        ~Baseline();
        void train(std::string saveFile);
        void loadSaved(std::string fname);

    private:
        float bogusMean();
        float betterMean();
        // Only used for better mean
        float globalAverage()
};

#endif // NAIVE_SVD_HPP
