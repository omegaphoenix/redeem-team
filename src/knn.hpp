#ifndef KNN_HPP
#define KNN_HPP
#include "model.hpp"

class kNN : public Model {
    public:
        kNN() : Model() {};
        ~kNN();
        void train();
    private:
        void pearson(float *x_i, float *x_j);
        void buildMatrix(float **train);
        void predict(int user, int movie);
        float **corrMatrix;
};

#endif // KNN_HPP

