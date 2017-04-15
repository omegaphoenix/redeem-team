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
        virtual std::pair<float, float> update(int user, int movie, float rating);
        float computeError();

        float **U;
        float **V;
        float eta;
        float lambda;
        int K;
        
};

#endif // NAIVE_SVD_HPP
