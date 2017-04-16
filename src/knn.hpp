#ifndef KNN_HPP
#define KNN_HPP
#include "model.hpp"
#include <vector>

class kNN : public Model {
    public:
        kNN() : Model() {};
        ~kNN();
        void train();
        void predict(int user, int movie);
    private:
        void pearson(float *x_i, float *x_j);
        void buildMatrix(std::vector<std::vector<float>> train, bool movie);
        std::vector<std::vector<float>> corrMatrix;
};

#endif // KNN_HPP

