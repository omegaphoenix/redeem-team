#ifndef TIME_SVD_PP_HPP
#define TIME_SVD_PP_HPP

#include <algorithm>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

#include "model.hpp"

using namespace std;

class TimeSVDPP : public Model {
    public:
        TimeSVDPP(float*,float*,int,float**,float**, string, string, string, string);
        ~TimeSVDPP();
        void sgd();
        float predictScore(float,int,int,int);
        float calcDev(int,int);    //calculate dev_u(t)
        int calcBin(int);    //calculate time bins
        float cValidate(float,float*,float*,float**,float**);
        void train(std::string saveFile);
        float predict(int user, int movie, int date);
    protected:

        //   prediction formula:
        //   avg + Bu + Bi
        //   + Bi_Bin,t + Alpha_u*Dev + Bu_t
        //   + Qi^T(Pu + |R(u)|^-1/2 \sum yi

        float* Tu; //variable for mean time of user
        float* Alpha_u;
        float* Bi;
        float** Bi_Bin;
        float* Bu;
        vector<map<int,float> > Bu_t;
        vector<map<int,float> > Dev; //save the result of calcDev(userId,time)
        float** Qi;
        float** Pu;
        float** y;
        float** sumMW;    //save the sum of Pu
        string trainFile;
        string crossFile;
        string testFile;
        string outFile;
        vector <pair <pair<int,int>, pair <int, int> > > test_data;
 };


#endif // TIME_SVD_PP_HPP
