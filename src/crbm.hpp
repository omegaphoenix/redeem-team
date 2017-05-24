#ifndef PURE_CRBM_HPP
#define PURE_CRBM_HPP
#include "model.hpp"

#define TOTAL_FEATURES  200
#define SOFTMAX         5
#define EPSILONW        0.00075   // Learning rate for weights
#define EPSILOND        0.000001  // Learning rate for weights
#define EPSILONVB       0.003     // Learning rate for biases of visible units
#define EPSILONHB       0.0003    // Learning rate for biases of hidden units
#define WEIGHTCOST      0.0001
#define MOMENTUM        0.84
#define FINAL_MOMENTUM   0.9
#define BATCH_SIZE   100
#define E  (0.00002) // stop condition
#define STD_DEV 0.01

class CRBM : public Model {

    public:
        CRBM();
        ~CRBM();
        void init();
        void train(std::string saveFile);
        void prepPredict(Model *mod, int n);
        float predict(int n, int i, int d = 0);
        void save(std::string fname);
        void loadSaved(std::string fname);

    private:
        // vishid are the weights.
        float vishid[N_MOVIES][SOFTMAX][TOTAL_FEATURES];
        float CDpos[N_MOVIES][TOTAL_FEATURES][SOFTMAX];
        float CDneg[N_MOVIES][TOTAL_FEATURES][SOFTMAX];
        float CDinc[N_MOVIES][TOTAL_FEATURES][SOFTMAX];
        float Dij[N_MOVIES][TOTAL_FEATURES];
        float DIJinc[N_MOVIES][TOTAL_FEATURES];

        float visbiases[N_MOVIES][SOFTMAX];
        float nvp2[N_MOVIES * SOFTMAX];
        float negvisprobs[N_MOVIES * SOFTMAX];
        float posvisact[N_MOVIES][SOFTMAX];
        float negvisact[N_MOVIES][SOFTMAX];
        float visbiasinc[N_MOVIES][SOFTMAX];

        unsigned int moviecount[N_MOVIES];
        unsigned char negvissoftmax[N_MOVIES];
        unsigned int movieseencount[N_MOVIES];

        float hidbiases[TOTAL_FEATURES];
        float poshidprobs[TOTAL_FEATURES];
        char  poshidstates[TOTAL_FEATURES];
        char  curposhidstates[TOTAL_FEATURES];
        float poshidact[TOTAL_FEATURES];
        float neghidact[TOTAL_FEATURES];
        float neghidprobs[TOTAL_FEATURES];
        char  neghidstates[TOTAL_FEATURES];
        float hidbiasinc[TOTAL_FEATURES];

        int prevUser, loopcount;

        Model *tester;
};
#endif // PURE_CRBM_HPP
