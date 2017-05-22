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
        TimeSVDPP(int,int,int,float*,float*,float*,float*,
                  float*,float*,float*,float*,
                  vector<map<int,float> >*,
                  vector<map<int,float> >*,
                  string,string,string);
        ~TimeSVDPP();
        void sgd();
        float predictScore(float,int,int,int);
        float calcDev(int,int);    //calculate dev_u(t)
        int calcBin(int);    //calculate time bins
        float cValidate(float);
        void train(std::string saveFile);
        void save(void);
        float predict(int user, int movie, int date);
    protected:
        int binNum;
        int factor;

        //   prediction formula:
        //   avg + Bu + Bi
        //   + Bi_Bin,t + Alpha_u*Dev + Bu_t
        //   + Qi^T(Pu + |R(u)|^-1/2 \sum yi

        vector<map<int,float> > Bu_t;
        vector<map<int,float> > Dev; //save the result of calcDev(userId,time)
        float* Tu; //variable for mean time of user
        float* Alpha_u;
        float* Bi;
        float* Bi_Bin;
        float* Bu;
        float* Qi;
        float* Pu;
        float* y;
        float* sumMW;    //save the sum of Pu
        string trainFile;
        string crossFile;
        string testFile;
        vector <pair <pair<int,int>, pair <int, int> > > test_data;
    private:
        string getBasename(void);
        string nickname;
        int numEpochs;
 };

TimeSVDPP* loadTSVDpp(string saveFile, string train_file,
                      string cross_file, string test_file);


#endif // TIME_SVD_PP_HPP
