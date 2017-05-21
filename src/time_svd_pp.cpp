#include "time_svd_pp.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <utility>

#include "utils.hpp"
#define sign(n) (n==0? 0 : (n<0?-1:1))    //define sign function

using namespace std;

const int binNum = 30;       //number of time bins
const float AVG = 3.60073;  //average score
float G_alpha = 0.00001;        //gamma for alpha
const float L_alpha = 0.0004;   //learning rate for alpha
const float L_pq = 0.015;       //learning rate for Pu & Qi
float G = 0.007;                //general gamma
const float Decay = 0.9;        //learning rate decay factor
const float L = 0.005;          //general learning rate
const int factor = 50;           //number of factors

//initialization
TimeSVDPP::TimeSVDPP(float* bi,float* bu,int k,float** qi,float** pu, string train_file, string cross_file, string test_file, string out_file):
    trainFile(train_file), crossFile(cross_file), testFile(test_file), outFile(out_file) {
    debugPrint("Initializing...\n");
    clock_t time0 = clock();

    train_data.resize(N_USERS);
    if (bi == NULL){
        Bi = new float[N_MOVIES];
        for (size_t i = 0; i < N_MOVIES; ++i){
            Bi[i] = 0.0;
        }
    }
    else {
        Bi = bi;
    }

    if (bu == NULL){
        Bu = new float[N_USERS];
        for (size_t i = 0;i < N_USERS; ++i){
            Bu[i] = 0.0;
        }
    }
    else{
        Bu = bu;
    }

    Alpha_u = new float[N_USERS];
    for (size_t i = 0; i < N_USERS; ++i){
        Alpha_u[i] = 0;
    }

    Bi_Bin = new float* [N_MOVIES];
    for (size_t i = 0; i < N_MOVIES; ++i){
        Bi_Bin[i] = new float[binNum];
    }

    for (size_t i = 0; i < N_MOVIES; ++i){
        for (size_t j = 0; j < binNum; ++j){
            Bi_Bin[i][j] = 0.0;
        }
    }


    if(qi == NULL){
        Qi = new float* [N_MOVIES];
        y = new float* [N_MOVIES];
        for (size_t i = 0; i < N_MOVIES; ++i){
            Qi[i] = new float[factor];
            y[i] = new float[factor];
        }

        for (size_t i = 0; i < N_MOVIES; ++i){
            for (size_t j = 0; j < factor; ++j){
                Qi[i][j] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(factor);
                y[i][j] = 0;
            }
        }
    }
    else{
        Qi = qi;
    }

    if(pu == NULL){
        sumMW = new float* [N_USERS];
        Pu = new float* [N_USERS];
        for (size_t i = 0; i < N_USERS; ++i){
            Pu[i] = new float[factor];
            sumMW[i] = new float[factor];
        }

        for (size_t i = 0; i < N_USERS; ++i){
            for (size_t j=0;j<factor;++j){
                sumMW[i][j] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(factor);
                Pu[i][j] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(factor);
            }
        }
    }   else{
        Pu = pu;
    }
    FILE *fp = fopen(trainFile.c_str(),"r");
    int userId,itemId,rating,t;
    while(fscanf(fp,"%d %d %d %d",&userId, &itemId, &t, &rating)!=EOF){
        train_data[userId - 1].push_back(make_pair(make_pair(itemId - 1,rating - 1),t - 1));
    }
    fclose(fp);
    fp = fopen(crossFile.c_str(),"r");
    while(fscanf(fp,"%d %d %d %d",&userId, &itemId, &t, &rating)!=EOF){
        test_data.push_back(make_pair(make_pair(userId - 1, itemId - 1),make_pair(t - 1,rating - 1)));
    }
    fclose(fp);

    Tu = new float[N_USERS];
    for (size_t i = 0;i<N_USERS;++i){
        float tmp = 0;
        if(train_data[i].size()==0)
        {
            Tu[i] = 0;
            continue;
        }
        for (size_t j=0;j<train_data[i].size();++j){
            tmp += train_data[i][j].second;
        }
        Tu[i] = tmp/train_data[i].size();
    }

    for (size_t i = 0;i<N_USERS;++i){
        map<int,float> tmp;
        for (size_t j=0;j<train_data[i].size();++j){
            if(tmp.count(train_data[i][j].second)==0)
            {
                tmp[train_data[i][j].second] = 0.0000001;
            }
            else continue;
        }
        Bu_t.push_back(tmp);
    }

    for (size_t i = 0;i<N_USERS;++i){
        map<int,float> tmp;
        Dev.push_back(tmp);
    }

    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Initializing took %f\n", ms1);
}

TimeSVDPP::~TimeSVDPP() {
    delete[] Bi;
    delete[] Bu;
    delete[] Alpha_u;
    delete[] Tu;
    for (size_t i = 0;i<N_USERS;++i){
        delete[] Pu[i];
        delete[] sumMW[i];
    }
    for (size_t i = 0;i<N_MOVIES;++i){
        delete[] Bi_Bin[i];
        delete[] Qi[i];
        delete[] y[i];
    }
    delete[] Bi_Bin;
    delete[] sumMW;
    delete[] y;
    delete[] Pu;
    delete[] Qi;
}

//calculate dev_u(t) = sign(t-tu)*|t-tu|^0.4 and save the result for saving the time
float TimeSVDPP::calcDev(int user, int timeArg) {
    if(Dev[user].count(timeArg)!=0) {
        return Dev[user][timeArg];
    }
    float tmp = sign(timeArg - Tu[user]) * pow(float(abs(timeArg - Tu[user])), 0.4);
    Dev[user][timeArg] = tmp;
    return tmp;
}

//calculate time bins
int TimeSVDPP::calcBin(int timeArg) {
    int binsize = N_DAYS/binNum + 1;
    return timeArg/binsize;
}

//main function for training
//terminate when RMSE varies less than 0.00005
void TimeSVDPP::train(std::string saveFile) {
    debugPrint("Training...\n");
    float preRmse = 1000;
    ofstream fout(outFile.c_str());
    srand(time(NULL));
    FILE *fp = fopen(testFile.c_str(),"r");
    int user, item, date, rating;
    float curRmse;
    for (size_t i = 0;i<2;++i) {
        sgd();
        curRmse = cValidate(AVG,Bu,Bi,Pu,Qi);
        cout << "test_Rmse in step " << i << ": " << curRmse << endl;
        if(curRmse >= preRmse-0.00005){
            break;
        }
        else{
            preRmse = curRmse;
        }
    }
    while (fscanf(fp,"%d %d %d %d",&user, &item, &date, &rating)!=EOF) {
        fout << predictScore(AVG, user - 1, item - 1, date - 1) << endl;
    }
    fclose(fp);
    fout.close();
    cout << "final RMSE: " << curRmse << endl;
}


//function for cross validation
float TimeSVDPP::cValidate(float avg,float* bu,float* bi,float** pu,float** qi){
    debugPrint("Cross validating...\n");
    clock_t time0 = clock();
    int userId,itemId,rating,t;
    int n = 0;
    float rmse = 0;
    for (const auto &ch:test_data){
        userId = ch.first.first;
        itemId = ch.first.second;
        t = ch.second.first;
        rating = ch.second.second;
        n++;
        float pScore = predictScore(avg,userId,itemId,t);
        rmse += (rating - pScore) * (rating - pScore);
    }
    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("Cross validation took %f ms\n", ms1);
    return sqrt(rmse/n);
}

float TimeSVDPP::predict(int user, int movie, int date) {
    return predictScore(AVG, user, movie, date);
}


//function for prediction
//   prediction formula:
//   avg + Bu + Bi
//   + Bi_Bin,t + Alpha_u*Dev + Bu_t
//   + Qi^T(Pu + |R(u)|^-1/2 \sum yi
float TimeSVDPP::predictScore(float avg,int userId, int itemId,int time){
    float tmp = 0.0;
    int sz = train_data[userId].size();
    float sqrtNum = 0;
    if (sz>1) sqrtNum = 1/(sqrt(sz));
    for (size_t i = 0;i<factor;++i){
        tmp += (Pu[userId][i] +sumMW[userId][i]*sqrtNum) * Qi[itemId][i];
    }
    float score = avg + Bu[userId] + Bi[itemId] + Bi_Bin[itemId][calcBin(time)] + Alpha_u[userId]*calcDev(userId,time) + Bu_t[userId][time] + tmp;

    if(score > 5){
        score = 5;
    }
    if(score < 1){
        score = 1;
    }
    return score;
}

//function for training
//update using stochastic gradient descent

void TimeSVDPP::sgd(){
    debugPrint("Updating using sgd...\n");
    clock_t time0 = clock();
    int userId,itemId,rating,time;
    for (userId = 0; userId < N_USERS; ++userId) {
        int sz = train_data[userId].size();
        float sqrtNum = 0;
        vector <float> tmpSum(factor,0);
        if (sz>1) sqrtNum = 1/(sqrt(sz));
        for (int k = 0; k < factor; ++k) {
            float sumy = 0;
            for (int i = 0; i < sz; ++i) {
                int itemI = train_data[userId][i].first.first;
                sumy += y[itemI][k];
            }
            sumMW[userId][k] = sumy;
        }
        for (int i = 0; i < sz; ++i) {
            itemId = train_data[userId][i].first.first;
            rating = train_data[userId][i].first.second;
            time = train_data[userId][i].second;
            float prediction = predictScore(AVG,userId,itemId,time);
            float error = rating - prediction;
            Bu[userId] += G * (error - L * Bu[userId]);
            Bi[itemId] += G * (error - L * Bi[itemId]);
            Bi_Bin[itemId][calcBin(time)] += G * (error - L * Bi_Bin[itemId][calcBin(time)]);
            Alpha_u[userId] += G_alpha * (error * calcDev(userId,time)  - L_alpha * Alpha_u[userId]);
            Bu_t[userId][time] += G * (error - L * Bu_t[userId][time]);

            for (size_t k=0;k<factor;k++){
                auto uf = Pu[userId][k];
                auto mf = Qi[itemId][k];
                Pu[userId][k] += G * (error * mf - L_pq * uf);
                Qi[itemId][k] += G * (error * (uf+sqrtNum*sumMW[userId][k]) - L_pq * mf);
                tmpSum[k] += error*sqrtNum*mf;
            }
        }
        for (int j = 0; j < sz; ++j) {
            itemId = train_data[userId][j].first.first;
            for (int k = 0; k < factor; ++k) {
                float tmpMW = y[itemId][k];
                y[itemId][k] += G*(tmpSum[k]- L_pq *tmpMW);
                sumMW[userId][k] += y[itemId][k] - tmpMW;
            }
        }
    }
    for (userId = 0; userId < N_USERS; ++userId) {
        auto sz = train_data[userId].size();
        float sqrtNum = 0;
        if (sz>1) sqrtNum = 1.0/sqrt(sz);
        for (int k = 0; k < factor; ++k) {
            float sumy = 0;
            for (int i = 0; i < sz; ++i) {
                int itemI = train_data[userId][i].first.first;
                sumy += y[itemI][k];
            }
            sumMW[userId][k] = sumy;
        }
    }
    G *= Decay;
    G_alpha *= Decay;
    debugPrint("Done updating using sgd...\n");
    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("SGD took %f ms\n", ms1);
}
