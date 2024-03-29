#ifndef MODEL_HPP
#define MODEL_HPP
#include <array>
#include <string>
#include <time.h>
#include <vector>
#include "utils.hpp"

class Model {
    public:
        Model();
        int* ratings; // COO format, movies and users 0-indexed
        unsigned int numRatings;
        unsigned char* values; // CSR values/ratings
        unsigned short* columns; // CSR columns/movies, 0-indexed
        unsigned short* dates; // CSR columns/movies, 0-indexed
        unsigned int* rowIndex; // CSR row index, where user i starts in values/columns, 0-indexed

        // MU variables
        int* muratings; // COO format
        unsigned char* muvalues; // CSR values/ratings
        int* mucolumns; // CSR columns/user
        int* mudates;
        int* murowIndex; // CSR row index, where user i starts in values/columns
        dataPoint* sortStruct;     

        virtual ~Model();
        void loadFresh(std::string inFname, std::string outFname);
        void loadCSR(std::string fname);
        void loadSaved(std::string fname);
        float validate(std::string valFile);
        float trainingError();
        void output(std::string saveFile, std::string valFile);
        void initLoad(std::string fname, std::string dataFile);
        void load(std::string dataFile);
#ifdef MU
        void transposeMU();
#endif

        virtual void train(std::string saveFile);
#ifdef ISRBM
        virtual void prepPredict(Model *mod, int n);
#endif
        virtual float predict(int n, int i, int d);
};

#endif // MODEL_HPP
