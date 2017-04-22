#ifndef RBM_HPP
#define RBM_HPP
#include "model.hpp"

#define N_FACTORS 100
#define MINIBATCH_SIZE 100
#define LEARNING_RATE 0.1
#define RBM_EPOCHS 5

class RBM : public Model {

    public:
        int N;
        int nVisible;
        int nHidden;
        double* hbias;
        double* vbias;
        RBM();
        ~RBM();
        void constrastiveDivergence(int*, double, int);
        void sampleHGivenV(int*, double*, int*);
        void sampleVGivenH(int*, double*, int*);
        double propUp(int*, double*, double);
        double propDown(int*, int, double);
        void gibbsHvh(int*, double*, int*, double*, int*);
        void reconstruct(int*, double*);
        void loadSaved(std::string fname) {};

		double sumOverFeatures(int movie, int rating, double* h);
		double** pCalcV(int** V, double* h, int user);
		void updateV(int** V, double** v, int user);
		int** createV(int user);
		double* pCalcH(int** V, int user);
		void updateH(double* h, int user, bool last, double threshold);
		void createMinibatch();
		void updateW(void);
		void train(std::string saveFile);

	private:
        double*** W;
		double** hidStates;
		int *minibatch;
		int *countUserRating; // number of movies rated
};
#endif // RBM_HPP
