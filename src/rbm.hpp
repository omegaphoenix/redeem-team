#ifndef RBM_HPP
#define RBM_HPP
#include <bitset>
#include "model.hpp"

#define N_FACTORS 100
#define MINIBATCH_SIZE 100
#define LEARNING_RATE 0.1
#define RBM_EPOCHS 100

class RBM : public Model {

    public:
        RBM();
        ~RBM();
        void init();
        void loadSaved(std::string fname) {};

        void setHidVar(int nthHidVar, bool newVal);
        bool getHidVar(int nthHidVar);
        void setV(int n, int i, int k, bool newVal);
        bool getV(int n, int i, int k);
        void updateW();
        void updateH();
        double getActualVal(int n, int i, int j, int k);
        double getExpectVal(int n, int i, int j, int k);
        void train(std::string saveFile) {};

    private:
        double* W;
        double* dW;
        double T;
        double epsilon; // learning rate
        double* hidBiases; // bias of feature j
        double* visBiases; // bias of rating k for movie i
        double* hidProbs; // hidden probabilities
        std::bitset<N_USERS * N_FACTORS> *hidVars;
        // We declare this so we don't have to reallocate memory each time
        std::bitset<N_MOVIES * MAX_RATING> *V;
};
#endif // RBM_HPP
