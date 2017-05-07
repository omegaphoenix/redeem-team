#ifndef RBM_HPP
#define RBM_HPP
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

        double sumOverFeatures(int movie, int rating, double* h);
        double** pCalcV(int** V, double* h, int user);
        void updateV(double** v, int user);
        int** createV(int user);
        void pCalcH(double* h, int** V, int user);
        void updateH(double* h, int user, bool last, double threshold);
        void createMinibatch();
        void updateW(void);
        void train(std::string saveFile);

    private:
        double*** W;
        double** hidStates;
        int* minibatch;
        int* countUserRating; // number of movies rated
};
#endif // RBM_HPP
