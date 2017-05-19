#ifndef KNN_HPP
#define KNN_HPP
#include "model.hpp"
#include <vector>

enum CorrelationMetric {
    kPearson = 0,
    kSpearman
};

class kNN : public Model {
    public:
        kNN();
        ~kNN();
        void loadSaved(std::string fname);
        float predict(int user, int movie);
        void train(std::string saveFile);
        void save(std::string fname);
        void normalizeRatings(double average_array[], double stdev_array[]);
        std::string getFilename(std::string data_file);
        float validate(int* valid_ratings, int valid_numRatings);
        int num_correlations;
        int baseline;
        CorrelationMetric metric;
        // ignore user pairs with fewer than shared_threshold movies in common
        int shared_threshold;
        // ignore users who have rated fewer than individual_threshold movies
        int individual_threshold;
        int K; // number of nearest neighbors
        double* avg_array;
        double* stdev_array;
    private:
        float pearson(int i_start, int i_end, int j_start, int j_end);
        float denormalize(float normalized, double stdev, double ave);
        void buildMatrix(std::string saveFile);
        float rmse(float actual, float predicted);
        int getRatingCSR(int user, int movie);
        float getDefaultRating(int user, double avg_array[]);
        std::vector<std::vector<float>> corrMatrix;
        float* normalized_values;
};

#endif // KNN_HPP

