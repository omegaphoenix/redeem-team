#ifndef RBM_HPP
#define RBM_HPP
#include <bitset>
#include "model.hpp"

#define N_FACTORS 100
#define RBM_EPOCHS 50
#define STD_DEV 0.01
#define BATCH_SIZE 1000

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
        void resetDeltas();
        void calcGrad(int startUser, int endUser);
        void posStep();
        void negStep();
        void updateW(int startUser, int endUser);
        void updateHidBias(int startUser, int endUser);
        void updateVisBias(int startUser, int endUser);
        void updateH();
        void updateV();
        void calcHidProbsUsingData();
        void calcHidProbs();
        void resetHidProbs();
        void compHidProbs();
        void calcVisProbs();
        void resetVisProbs();
        void sumVisProbs();
        void sumToVisProbs();
        float getActualVal(int n, int i, int j, int k);
        float getExpectVal(int n, int i, int j, int k);
        void runGibbsSampler();
        void train(std::string saveFile);
        float predict(int n, int i);

    private:
        float* W;
        float* dW;
        float* incW;
        float* hidBiases; // bias of feature j
        float* dHidBiases; // delta of bias of feature j
        float* incHidBiases;
        float* visBiases; // bias of rating k for movie i
        float* dVisBiases; // delta of bias of rating k for movie i
        float* incVisBiases;
        float* hidProbs; // hidden probabilities
        float* visProbs; // visible probabilities
        std::bitset<N_USERS * N_FACTORS>* hidVars; // bitset
        std::bitset<N_MOVIES * MAX_RATING>* indicatorV; // array of bitsets
        float eta; // momentum
        float epsilon; // learning rate
        float lambda; // weight decay
        unsigned int T; // See equation 6 in RBM for CF, Salakhutdinov 2007
};
#endif // RBM_HPP
