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
#define BATCH_SIZE   100
#define E  (0.00002) // stop condition
#define STD_DEV 0.01

class RBM : public Model {

    public:
        RBM();
        ~RBM();
        void init();
        void train(std::string saveFile);
        float predict(int n, int i);

    private:
        // vishid are the weights.
        float vishid[N_MOVIES][SOFTMAX][TOTAL_FEATURES];
        float visbiases[N_MOVIES][SOFTMAX];
        float hidbiases[TOTAL_FEATURES];
        float CDpos[N_MOVIES][SOFTMAX][TOTAL_FEATURES];
        float CDneg[N_MOVIES][SOFTMAX][TOTAL_FEATURES];
        float CDinc[N_MOVIES][SOFTMAX][TOTAL_FEATURES];

        float poshidprobs[TOTAL_FEATURES];
        char   poshidstates[TOTAL_FEATURES];
        char   curposhidstates[TOTAL_FEATURES];
        float poshidact[TOTAL_FEATURES];
        float neghidact[TOTAL_FEATURES];
        float neghidprobs[TOTAL_FEATURES];
        char   neghidstates[TOTAL_FEATURES];
        float hidbiasinc[TOTAL_FEATURES];

        float nvp2[N_MOVIES][SOFTMAX];
        float negvisprobs[N_MOVIES][SOFTMAX];
        unsigned char   negvissoftmax[N_MOVIES];
        float posvisact[N_MOVIES][SOFTMAX];
        float negvisact[N_MOVIES][SOFTMAX];
        float visbiasinc[N_MOVIES][SOFTMAX];

        unsigned int moviecount[N_MOVIES];
        int prevUser;

};
#endif // PURE_RBM_HPP
