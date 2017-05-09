#ifndef RBM_HPP
#define RBM_HPP
#include <bitset>
#include "model.hpp"

#define N_FACTORS 10
#define RBM_EPOCHS 1

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
        void calcGrad();
        void posStep();
        void negStep();
        void updateW();
        void updateH();
        void updateV();
        void calcHidProbsUsingData();
        void calcHidProbs();
        void calcVisProbs();
        double getActualVal(int n, int i, int j, int k);
        double getExpectVal(int n, int i, int j, int k);
        void runGibbsSampler();
        void train(std::string saveFile);

    private:
        double* W;
        double* dW; // So we don't have to reallocate each time.
        unsigned int T; // See equation 6 in RBM for CF, Salakhutdinov 2007
        double epsilonW; // learning rate for weights
        double epsilonVB; // learning rate for biases of visible units
        double epsilonHB; // learning rate for biases of hidden units
        double* hidBiases; // bias of feature j
        double* dHidBiases; // delta of bias of feature j
        double* visBiases; // bias of rating k for movie i
        double* dVisBiases; // delta of bias of rating k for movie i
        double* hidProbs; // hidden probabilities
        double* visProbs; // visible probabilities
        std::bitset<N_USERS * N_FACTORS>* hidVars; // bitset
        std::bitset<N_MOVIES * MAX_RATING>* indicatorV; // array of bitsets
};
#endif // RBM_HPP
