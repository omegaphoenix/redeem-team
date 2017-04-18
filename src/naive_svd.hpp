#ifndef NAIVE_SVD_HPP
#define NAIVE_SVD_HPP
#include <string>
#include <utility>
#include "model.hpp"

class NaiveSVD : public Model {
    public:
        NaiveSVD();
        ~NaiveSVD();
        void setParams(int K, float eta, float lambda);
        void train(std::string saveFile);
        float validate(std::string valFile, std::string saveFile);
        void save(std::string fname);
        void loadSaved(std::string fname);
        void printOutput(std::string fname);
        int numEpochs;
        int MAX_EPOCHS;
    private:
        float runEpoch();
        virtual void update(int user, int movie, float rating);
        float computeError();
        float dotProduct(int user, int movie);

        float *U;
        float *V;
        float eta;
        float lambda;
        int K;
        float EPSILON;

        bool validation_loaded;
};

#endif // NAIVE_SVD_HPP
