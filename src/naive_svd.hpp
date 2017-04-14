#ifndef NAIVE_SVD_HPP
#define NAIVE_SVD_HPP
#include "model.hpp"

class NaiveSVD : public Model {
    public:
        NaiveSVD() : Model() {};
        ~NaiveSVD();
        void train();
    private:
        virtual void update();
        float **U;
        float **V;
};

#endif // NAIVE_SVD_HPP
