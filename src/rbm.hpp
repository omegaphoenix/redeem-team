#ifndef RBM_HPP
#define RBM_HPP
#include "model.hpp"

class RBM : public Model {

    public:
        int N;
        int nVisible;
        int nHidden;
        double **W;
        double *hbias;
        double *vbias;
        RBM(int, int, int, double**, double*, double*);
        ~RBM();
        void constrastiveDivergence(int*, double, int);
        void sampleHGivenV(int*, double*, int*);
        void sampleVGivenH(int*, double*, int*);
        double propUp(int*, double*, double);
        double propDown(int*, int, double);
        void gibbsHvh(int*, double*, int*, double*, int*);
        void reconstruct(int*, double*);
        void loadSaved(std::string fname) {};
        void train(std::string saveFile) {};
};
#endif // RBM_HPP
