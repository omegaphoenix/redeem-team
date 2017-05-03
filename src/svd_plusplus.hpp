#ifndef NAIVE_SVD_HPP
#define NAIVE_SVD_HPP
#include <string>
#include <utility>
#include "model.hpp"
#include "baseline.hpp"

class SVDPlus : public Model {
    public:
        SVDPlus();
        ~SVDPlus();
        void setParams(int K, float eta, float lambda, 
            float mu, float* user_bias, float* movie_bias);
        void train(std::string saveFile);
        float validate(std::string valFile, std::string saveFile);
        void save(std::string fname);
        void loadSaved(std::string fname);
        void printOutput(std::string fname);
        int numEpochs;
        int MAX_EPOCHS;
    private:
        float runEpoch();
        virtual void update(int user, int movie, float rating, 
            float* sum_y, float e_ui, bool update_yw);
        float computeAllError();
        float dotProduct(int user, int movie);
        // TODO: write these functions
        void getPlusVariables(int user, int movie, float N, 
            float* sum_y, float &sum_w, float &sum_c);
        float getBias(int user, int movie);
        float predictRating(int user, int movie, 
            float* sum_y, float sum_w, float sum_c);

        float *U;
        float *V;
        float eta;
        float lambda;
        int K;
        float EPSILON;

        // Baseline variables
        // TODO: initialize these values.
        float mu;
        float *user_bias;
        float *movie_bias;

        // SVD++ variables
        // TODO: initialize these arrays.
        float *Y;
        float *C;
        float *W;

        bool validation_loaded;
};

#endif // NAIVE_SVD_HPP
