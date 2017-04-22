#include "rbm.hpp"
#include <cmath>
#include <iostream>
#include <math.h>
#include <time.h>
using namespace std;

// Initialize RBM variables.
RBM::RBM() {
    clock_t time0 = clock();
    printf("Initializing RBM...\n");

    // Initialize W
    this->W = new double**[N_MOVIES];

    for (unsigned int i = 0; i < N_MOVIES; ++i) {
        this->W[i] = new double*[N_FACTORS];
        for (unsigned int j = 0; j < N_FACTORS; ++j) {
            this->W[i][j] = new double[MAX_RATING];
        }
    }

    this->hidStates = new double*[N_USERS];
    for (unsigned int i = 0; i < N_USERS; ++i) {
        this->hidStates[i] = new double[N_FACTORS];
    }

    this->minibatch = new int[MINIBATCH_SIZE];

    this->countUserRating = new int[N_USERS];
    for (unsigned int i = 0; i < N_USERS; ++i) {
        countUserRating[i] = rowIndex[i + 1] - rowIndex[i];
    }

    clock_t time1 = clock();
    double ms1 = diffclock(time1, time0);
    std::cout << "RBM initialization took " << ms1 << " ms" << std::endl;
}

RBM::~RBM() {
    delete[] this->minibatch;
    delete[] this->countUserRating;

    for(unsigned int i = 0; i < N_MOVIES; ++i) {
        for (unsigned int j = 0; j < N_FACTORS; ++j) {
            delete[] this->W[i][j];
        }
        delete[] this->W[i];
    }

    for (unsigned int i = 0; i < N_USERS; ++i) {
        delete[] this->hidStates[i];
    }

    delete[] this->W;
	delete[] this->hidStates;
}

// movie is 0-indexed
double RBM::sumOverFeatures(int movie, int rating, double* h) {
	double total = 0;
    for (unsigned int i = 0; i < N_FACTORS; ++i) {
		// ratings are indexed 0-4
        total += h[i] * this->W[movie][i][rating - 1];
    }
	return total;
}

// Return expected value for user.
double** RBM::pCalcV(int** V, double* h, int user) {
    int index = rowIndex[user];
    int count = this->countUserRating[user];
    int movie, eValue;
    double numer, denom = 0;
    // Determine most likely
    for (unsigned int i = 0; i < count; ++i) {
        movie = columns[index + i];
        for (unsigned int j = 1; j <= MAX_RATING; ++j) {
            numer = exp(sumOverFeatures(movie, j, h));
            for (unsigned int k = 1; k <= MAX_RATING; ++k) {
                denom += exp(sumOverFeatures(movie, k, h));
            }
            V[i][j] = numer / denom;
        }
    }
    // Stored as movie count x 2 array
    double** v = new double*[count];
    for (unsigned int i = 0; i < count; ++i) {
        v[i] = new double[2];
        v[i][0] = V[i][0];
        eValue = V[i][1] + (2*V[i][2]) + (3*V[i][3])+ (4*V[i][4])+ (5*V[i][5]);
        v[i][1] = eValue;
    }
    return v;
}

// Update v.
void RBM::updateV(int** V, double** v, int user) {
    int count = this->countUserRating[user];
    for(unsigned int i = 0; i < count; ++i) {
        V[i][1] = bound(v[i][1]);
    }
}

// Create V.
int** RBM::createV(int user) {
    int index = rowIndex[user];
    int count = this->countUserRating[user];
    int movie, rating;
    int** V = new int*[count];
    // Fill up V with movies/ratings
    for (unsigned int i = 0; i < count; ++i) {
        // Initialize array
        V[i] = new int[MAX_RATING + 1];
        movie = columns[index + i];
        rating = values[index + i];
        V[i][0] = movie;
        V[i][rating - 1] = 1;
    }
    return V;
}

// Fill up h with appropriate weight probabilities for each user.
double* RBM::pCalcH(int** V, int user) {
    double* h = new double[N_FACTORS];
    int term, movie, rating; 
    int count = this->countUserRating[user];
    for (unsigned int i = 0; i < N_FACTORS; ++i) {
        term = 0;
        for (unsigned int j = 0; j < count; ++j) {
            movie = V[j][0];
            rating = V[j][1];
            term += this->W[movie][i][rating - 1];
        }
        h[i] = 1/(1 + exp(-1 * term));
    }
    return h;
}

// Update h for each user.
void RBM::updateH(double* h, int user, bool last, double threshold) {
    // Update h
    if (!last) {
        for (unsigned int i = 0; i < N_FACTORS; ++i) {
            if (h[i] > threshold) {
                h[i] = 1;
            }
            else {
                h[i] = 0;
            }
        }
    }
}

void RBM::createMinibatch() {
    unsigned int val = 0;
    for (unsigned int i = 0; i < MINIBATCH_SIZE; ++i) {
        val = minibatchRandom();
        minibatch[i] = val;
    }
}

void RBM::updateW() {
    int** V;
    double **v;
    int user, size;
    // Initialize
    double*** expData = new double**[N_MOVIES];
    double*** expRecon = new double**[N_MOVIES];
    for (unsigned int i = 0; i < N_MOVIES; ++i) {
        expData[i] = new double*[N_FACTORS];
        expRecon[i] = new double*[N_FACTORS];
        for (unsigned int j = 0; j < N_FACTORS; ++j) {
            expData[i][j] = new double[MAX_RATING];
            expRecon[i][j] = new double[MAX_RATING];
        }
    }

    for (unsigned int i = 0; i < MINIBATCH_SIZE; ++i) {
        user = this->minibatch[i];
        V = createV(user);
        size = this->countUserRating[user];
        this->hidStates[user] = pCalcH(V, user);
        updateH(this->hidStates[user], user, false, oneRand());
        for (unsigned int j = 0; j < size; ++j) {
            for (unsigned int k = 0; k < N_FACTORS; ++k) {
                expData[V[j][0]][k][V[j][1] - 1] += this->hidStates[user][k];
            }
        }
        v = pCalcV(V, this->hidStates[user], user);
        updateV(V, v, user);
        this->hidStates[user] = pCalcH(V, user);
        updateH(this->hidStates[user], user, false, oneRand());
        for (unsigned int j = 0; j < size; ++j) {
            for (unsigned int k = 0; k < N_FACTORS; ++k) {
                expRecon[V[j][0]][k][V[j][1] - 1] += this->hidStates[user][k];
            }
        }
    }

    // Update W
    matrixAdd(expData, expRecon, N_MOVIES, N_FACTORS, MAX_RATING, -1);
    matrixScalarMult(expData, (LEARNING_RATE / size), N_MOVIES, N_FACTORS, MAX_RATING);
    matrixAdd(W, expData, N_MOVIES, N_FACTORS, MAX_RATING, 1);

    // Delete all pointer arrays
    for (unsigned int i = 0; i < size; ++i) {
        delete[] v[i];
        delete[] V[i];
    }

    for(unsigned int i = 0; i < N_MOVIES; ++i) {
        for (unsigned int j = 0; j < N_FACTORS; ++j) {
            delete[] expData[i][j];
            delete[] expRecon[i][j];
        }
        delete[] expData[i];
        delete[] expRecon[i];
    }

    delete[] expData;
    delete[] expRecon;
    delete[] v;
    delete[] V;
}

void RBM::train(std::string saveFile) {
    int user, rating, movie, predict, err, trainCount, trainErr, numer, denom;
    clock_t start, end;

    for (unsigned int i = 0; i < RBM_EPOCHS; ++i) {
        start = clock();
        printf("Epoch Number: %d.\n", i);
        createMinibatch();
        updateW();
        trainErr = 0;
        trainCount = 0;

        if (i % 100 == 0 && i != 0) {
            for(unsigned int j = 0; j < numRatings; ++j) {
                user = ratings[j * DATA_POINT_SIZE + USER_IDX];
                movie = ratings[j * DATA_POINT_SIZE + MOVIE_IDX];
                rating = ratings[j * DATA_POINT_SIZE + RATING_IDX];
                predict = 0;
                numer = 0;
                denom = 0;

                for (unsigned int k = 1; k <= MAX_RATING; ++k) {
                    numer = exp(sumOverFeatures(movie, k, this->hidStates[user]));
                    for (unsigned int l = 1; l <= MAX_RATING; ++l) {
                        denom += exp(sumOverFeatures(movie, l, this->hidStates[user]));
                    }
                    predict += (numer / denom) * k;
                }

                predict = bound(predict);

                err = (double) rating - predict;

                trainErr += err * err;

                trainCount++;
            }

            end = clock();
            printf("Train RMSE: %f. Took %.f ms.\n",
                    sqrt(trainErr / (double) numRatings), diffclock(end, start));
        } else {
            end = clock();
            printf("Took %.f ms.\n", diffclock(end, start));
        }
    }
}

int main() {
    // Speed up stdio operations
    std::ios_base::sync_with_stdio(false);
    srand(0);

    clock_t time0 = clock();
    // Initialize
    RBM *rbm = new RBM();
    clock_t time1 = clock();
    rbm->load("1.dta");
    clock_t time2 = clock();

    // Learn parameters
    rbm->train("data/um/LebronCanSuckMy5Rings.save");
    clock_t time3 = clock();
    double ms1 = diffclock(time1, time0);
    std::cout << "Initializing took " << ms1 << " ms" << std::endl;
    double ms2 = diffclock(time2, time1);
    std::cout << "Total loading took " << ms2 << " ms" << std::endl;
    double ms3 = diffclock(time3, time2);
    std::cout << "Training took " << ms3 << " ms" << std::endl;
    double total_ms = diffclock(time3, time0);
    std::cout << "Total running time was " << total_ms << " ms" << std::endl;
    return 0;
}
