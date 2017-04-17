#ifndef NAIVE_SVD_HPP
#define NAIVE_SVD_HPP
#include <string>
#include <utility>
#include "model.hpp"

class NaiveSVD : public Model {
    public:
        NaiveSVD() : Model() {};
        ~NaiveSVD();
        void setParams(int K, float eta, float lambda);
        void train();
        void save(std::string fname);
        void loadSaved(std::string fname);
        void printOutput(std::string fname);
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
        int MAX_EPOCHS;
        float EPSILON;
        
};

#endif // NAIVE_SVD_HPP
