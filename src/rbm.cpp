#include "rbm.hpp"
#include <assert.h>
#include <cmath>
#include <float.h>
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
            for (unsigned int k = 0; k < MAX_RATING; ++k) {
                this->W[i][j][k] = normalRandom() * 0.1;
            }
        }
    }

    this->hidStates = new double*[N_USERS];
    for (unsigned int i = 0; i < N_USERS; ++i) {
        this->hidStates[i] = new double[N_FACTORS];
        for (unsigned int j = 0; j < N_FACTORS; ++j) {
            this->hidStates[i][j] = normalRandom() * 0.1;
        }
    }

    this->minibatch = new int[MINIBATCH_SIZE];

    this->countUserRating = new int[N_USERS];

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

void RBM::init() {
    this->load("1.dta");
    int userNumRatings = 0;
    for (unsigned int i = 0; i < N_USERS; ++i) {
        userNumRatings = rowIndex[i + 1] - rowIndex[i];
        assert(userNumRatings >= 0 && userNumRatings <= N_MOVIES);
        this->countUserRating[i] = userNumRatings;
    }
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
    // Stored as movie count x 2 array
    double** temp = new double*[count];
    // Determine most likely
    for (unsigned int i = 0; i < count; ++i) {
        movie = columns[index + i];
        assert(V[i][0] == movie);
        temp[i] = new double[MAX_RATING + 1];
        for (unsigned int j = 1; j <= MAX_RATING; ++j) {
            numer = exp(sumOverFeatures(movie, j, h));
            assert(numer >= 0);
            for (unsigned int k = 1; k <= MAX_RATING; ++k) {
                denom += exp(sumOverFeatures(movie, k, h));
            }
            double prob = numer /denom;
            for (unsigned int l = 0; l < N_FACTORS; ++l) {
                cout << "hidStates" << l << ": " << this->hidStates[user][l] << endl;
            }
            assert(prob >= 0 && prob <= 1);
            temp[i][j] = prob;
        }
    }
    double** v = new double*[count];
    for (unsigned int i = 0; i < count; ++i) {
        v[i] = new double[2];
        v[i][0] = V[i][0];
        eValue = temp[i][1] + (2*temp[i][2]) + (3*temp[i][3])+ (4*temp[i][4])+ (5*temp[i][5]);
        v[i][1] = eValue;
        delete[] temp[i];
    }
    delete[] temp;
    return v;
}

// Update v.
void RBM::updateV(double** v, int user) {
    int count = this->countUserRating[user];
    for(unsigned int i = 0; i < count; ++i) {
        v[i][1] = bound(v[i][1]);
    }
}

// Create V.
int** RBM::createV(int user) {
    assert(user > 0 && user < N_USERS);
    int index = rowIndex[user];
    int count = this->countUserRating[user];
    int movie, rating;
    int** newV = new int*[count];
    // Fill up V with movies/ratings
    for (unsigned int i = 0; i < count; ++i) {
        // Initialize array
        newV[i] = new int[MAX_RATING + 1]();
        movie = columns[index + i];
        rating = values[index + i];
        newV[i][0] = movie;
        newV[i][rating] = 1;
    }
    return newV;
}

// Fill up h with appropriate weight probabilities for each user.
void RBM::pCalcH(double* h, int** V, int user) {
    int term, movie;
    int count = this->countUserRating[user];
    for (unsigned int i = 0; i < N_FACTORS; ++i) {
        term = 0;
        for (unsigned int j = 0; j < count; ++j) {
            movie = V[j][0];
            for (unsigned int k = 0; k < MAX_RATING; ++k) {
                term += this->W[movie][i][k] * V[j][k+1];
            }
        }
        h[i] = 1/(1 + exp(-1 * term));
    }
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
        debugPrint("Creating V\n");
        int** V = createV(user);
        size = this->countUserRating[user];
        debugPrint("Calculating hidden states\n");
        pCalcH(this->hidStates[user], V, user);
        debugPrint("Update hidden states\n");
        updateH(this->hidStates[user], user, false, oneRand());
        debugPrint("Update hidden states part 2\n");
        for (unsigned int j = 0; j < size; ++j) {
            int movie = V[j][0];
            for (unsigned int k = 0; k < N_FACTORS; ++k) {
                for (unsigned int l = 0; l < MAX_RATING; ++l) {
                    expData[movie][k][l] += this->hidStates[user][k] * V[j][l+1];
                }
            }
        }
        debugPrint("Calculating V\n");
        double **v = pCalcV(V, this->hidStates[user], user);
        debugPrint("Updating V\n");
        updateV(v, user);
        pCalcH(this->hidStates[user], V, user);
        debugPrint("Updating H\n");
        updateH(this->hidStates[user], user, false, oneRand());
        for (unsigned int j = 0; j < size; ++j) {
            int movie = V[j][0];
            for (unsigned int k = 0; k < N_FACTORS; ++k) {
                for (unsigned int l = 0; l < MAX_RATING; ++l) {
                    expRecon[movie][k][l] += this->hidStates[user][k] * V[j][l+1];
                }
            }
        }
        debugPrint("Deleting parts of V\n");
        for (unsigned int j = 0; j < size; ++j) {
            delete[] V[j];
            delete[] v[j];
        }
        debugPrint("Deleting V\n");
        delete[] V;
        delete[] v;
    }

    // Update W
    matrixAdd(expData, expRecon, N_MOVIES, N_FACTORS, MAX_RATING, -1);
    matrixScalarMult(expData, (LEARNING_RATE / size), N_MOVIES, N_FACTORS, MAX_RATING);
    matrixAdd(W, expData, N_MOVIES, N_FACTORS, MAX_RATING, 1);

    // Clean up memory
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
}

void RBM::train(std::string saveFile) {
    int user, rating, movie, predict, err, trainCount, trainErr, numer, denom;
    clock_t start, end;
    double RMSE, timeElapsed;

    FILE* out;
    for (unsigned int i = 0; i < RBM_EPOCHS; ++i) {
        start = clock();
        printf("Epoch Number: %d.\n", i);
        createMinibatch();
        debugPrint("Updating W\n");
        updateW();
        debugPrint("Finished updating W\n");
        trainErr = 0;
        trainCount = 0;

        if ((i + 1) % 100 == 0) {
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
                    for (unsigned int l = 0; l < N_FACTORS; ++l) {
                        cout << "hidStates" << l << ": " << this->hidStates[user][l] << endl;
                    }
                    predict += (numer / denom) * k;
                }

                cout << "prediction before bounding: " << predict << endl;
                predict = bound(predict);
                cout << "prediction: " << predict << endl;

                err = (double) rating - predict;

                trainErr += err * err;

                trainCount++;
                break;
            }

            end = clock();
            RMSE = sqrt(trainErr / (double) numRatings);
            timeElapsed = diffclock(end, start);
            printf("Train RMSE: %f. Took %.f ms.\n", RMSE, timeElapsed);
            out = fopen((saveFile).c_str(), "a");
            fprintf(out, "Train RMSE: %f. Took %.f ms.\n", RMSE, timeElapsed);
            fclose(out);
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
    rbm->init();
    clock_t time2 = clock();

    // Learn parameters
    rbm->train("data/um/rbm.save");
    clock_t time3 = clock();
    double ms1 = diffclock(time1, time0);
    std::cout << "Initializing took " << ms1 << " ms" << std::endl;
    double ms2 = diffclock(time2, time1);
    std::cout << "Total loading took " << ms2 << " ms" << std::endl;
    double ms3 = diffclock(time3, time2);
    std::cout << "Training took " << ms3 << " ms" << std::endl;
    double total_ms = diffclock(time3, time0);
    std::cout << "Total running time was " << total_ms << " ms" << std::endl;
    delete rbm;
    return 0;
}
