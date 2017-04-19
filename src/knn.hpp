#ifndef KNN_HPP
#define KNN_HPP
#include "model.hpp"
#include <vector>

class kNN : public Model {
    public:
        kNN() : Model() {};
        ~kNN();
        void loadSaved(std::string fname);
        void predict(int user, int movie);
        void train(std::string saveFile);
        void save(std::string fname);
        int num_correlations;
    private:
        float pearson(int i_start, int i_end, int j_start, int j_end);
        void buildMatrix();
        std::vector<std::vector<float>> corrMatrix;
};

#endif // KNN_HPP

