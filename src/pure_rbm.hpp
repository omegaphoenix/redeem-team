#ifndef PURE_RBM_HPP
#define PURE_RBM_HPP
#include "model.hpp"

#define TOTAL_FEATURES  100
#define SOFTMAX         5
#define EPSILONW        0.001   // Learning rate for weights
#define EPSILONVB       0.008   // Learning rate for biases of visible units
#define EPSILONHB       0.0006  // Learning rate for biases of hidden units
#define WEIGHTCOST      0.0001
#define MOMENTUM        0.8
#define FINAL_MOMENTUM   0.9
#define E  (0.00002) // stop condition

class RBM : public Model {

    public:
        RBM();
        ~RBM();
        void init();
        void train(std::string saveFile);
        float predict(int n, int i);

    private:
        // vishid are the weights.
        double vishid[N_MOVIES][SOFTMAX][TOTAL_FEATURES];
        double visbiases[N_MOVIES][SOFTMAX];
        double hidbiases[TOTAL_FEATURES];
        double CDpos[N_MOVIES][SOFTMAX][TOTAL_FEATURES];
        double CDneg[N_MOVIES][SOFTMAX][TOTAL_FEATURES];
        double CDinc[N_MOVIES][SOFTMAX][TOTAL_FEATURES];

        double poshidprobs[TOTAL_FEATURES];
        char   poshidstates[TOTAL_FEATURES];
        char   curposhidstates[TOTAL_FEATURES];
        double poshidact[TOTAL_FEATURES];
        double neghidact[TOTAL_FEATURES];
        double neghidprobs[TOTAL_FEATURES];
        char   neghidstates[TOTAL_FEATURES];
        double hidbiasinc[TOTAL_FEATURES];

        double nvp2[N_MOVIES][SOFTMAX];
        double negvisprobs[N_MOVIES][SOFTMAX];
        char   negvissoftmax[N_MOVIES];
        double posvisact[N_MOVIES][SOFTMAX];
        double negvisact[N_MOVIES][SOFTMAX];
        double visbiasinc[N_MOVIES][SOFTMAX];

        unsigned int moviercount[SOFTMAX*N_MOVIES];
        unsigned int moviecount[N_MOVIES];

};
#endif // PURE_RBM_HPP
