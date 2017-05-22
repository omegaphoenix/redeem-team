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

using namespace std;

const int kBinNum = 30;         // number of time bins
const float AVG = 3.60073;     // average score
float G_alpha = 0.00001;       // gamma for alpha
const float L_alpha = 0.0004;  // learning rate for alpha
const float L_pq = 0.015;      // learning rate for Pu & Qi
float G = 0.007;               // general gamma
const float Decay = 0.9;       // learning rate decay factor
const float L = 0.005;         // general learning rate
const int kFactor = 50;         // number of factors

//initialization
TimeSVDPP::TimeSVDPP(int epochs, int bnum, int k, float* alpha_u, float* bi,
        float* bi_bin, float* bu, float* qi, float* pu, float* ys, float* summw,
        vector<map<int,float> >* bu_t, vector<map<int,float> >* dev,
        string train_file, string cross_file, string test_file, string out_file):
    trainFile(train_file), crossFile(cross_file), testFile(test_file), outFile(out_file) {
    debugPrint("Initializing...\n");
    clock_t time0 = clock();

    numEpochs = epochs;

    if (bnum == 0) {
        binNum = kBinNum;
    }
    else {
        binNum = bnum;
    }

    if (k == 0) {
        factor = kFactor;
    }
    else {
        factor = k;
    }

    if (bi == NULL) {
        Bi = new float[N_MOVIES];
        for (size_t i = 0; i < N_MOVIES; ++i) {
            Bi[i] = 0.0;
        }
    }
    else {
        Bi = bi;
    }

    if (bu == NULL) {
        Bu = new float[N_USERS];
        for (size_t i = 0; i < N_USERS; ++i) {
            Bu[i] = 0.0;
        }
    }
    else{
        Bu = bu;
    }

    if (alpha_u == NULL) {
        Alpha_u = new float[N_USERS];
        for (size_t i = 0; i < N_USERS; ++i) {
            Alpha_u[i] = 0;
        }
    }
    else {
        Alpha_u = alpha_u;
    }

    if (bi_bin == NULL) {
        Bi_Bin = new float[N_MOVIES * binNum];
        for (size_t i = 0; i < N_MOVIES * binNum; ++i) {
            Bi_Bin[i] = 0.0;
        }
    }
    else {
        Bi_Bin = bi_bin;
    }

    if(qi == NULL) {
        Qi = new float[N_MOVIES * factor];
        y = new float[N_MOVIES * factor];
        for (size_t i = 0; i < N_MOVIES; ++i) {
            for (size_t j = 0; j < factor; ++j) {
                Qi[i * factor + j] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(factor);
                y[i * factor + j] = 0;
            }
        }
    }
    else{
        Qi = qi;
        y = ys;
    }

    if(pu == NULL) {
        sumMW = new float[N_USERS * factor];
        Pu = new float[N_USERS * factor];
        for (size_t i = 0; i < N_USERS; ++i) {
            for (size_t j = 0; j<factor; ++j) {
                sumMW[i * factor + j] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(factor);
                Pu[i * factor + j] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(factor);
            }
        }
    }   else{
        Pu = pu;
        sumMW = summw;
    }
    debugPrint("Loading data...\n");
    clock_t time1 = clock();
    int userId,itemId,rating,t;
    load(trainFile);
    FILE *fp = fopen(crossFile.c_str(),"r");
    while(fscanf(fp,"%d %d %d %d",&userId, &itemId, &t, &rating) != EOF) {
        test_data.push_back(make_pair(make_pair(userId - 1, itemId - 1),make_pair(t - 1,rating)));
    }
    fclose(fp);
    clock_t time2 = clock();
    float ms1 = diffclock(time2, time1);
    printf("Loading data took %f\n", ms1);

    Tu = new float[N_USERS];
    for (size_t i = 0; i < N_USERS; ++i) {
        float tmp = 0;
        int userEnd = rowIndex[i + 1];
        int userStart = rowIndex[i];
        int sz = userEnd - userStart;
        if(sz == 0) {
            Tu[i] = 0;
            continue;
        }
        for (size_t dateIdx = userStart; dateIdx < userEnd; ++dateIdx) {
            tmp += dates[dateIdx];
        }
        Tu[i] = tmp/sz;
    }

    if (bu_t == NULL) {
        for (size_t i = 0; i < N_USERS; ++i) {
            map<int,float> tmp;
            int userEnd = rowIndex[i + 1];
            int userStart = rowIndex[i];
            for (size_t dateIdx = userStart; dateIdx < userEnd; ++dateIdx) {
                int date = dates[dateIdx];
                if(tmp.count(date) == 0) {
                    tmp[date] = 0.0000001;
                }
            }
            Bu_t.push_back(tmp);
        }
    }
    else {
        Bu_t = *bu_t;
    }

    if (dev == NULL) {
        for (size_t i = 0; i < N_USERS; ++i) {
            map<int,float> tmp;
            Dev.push_back(tmp);
        }
    }
    else {
        Dev = *dev;
    }

    clock_t time3 = clock();
    float ms2 = diffclock(time3, time0);
    printf("Initializing took %f\n", ms2);
}

TimeSVDPP::~TimeSVDPP() {
    delete[] Tu;
    delete[] Alpha_u;
    delete[] Bi;
    delete[] Bi_Bin;
    delete[] Bu;
    delete[] Qi;
    delete[] Pu;
    delete[] y;
    delete[] sumMW;
}

// Save progress
void TimeSVDPP::save(string nickname) {
    if (nickname != "") {
        nickname += "_";
    }
    string fname = "model/timesvdpp/" + nickname +
                   to_string(factor) + "factors_" +
                   to_string(binNum) + "bins_" +
                   to_string(G) + "G_" +
                   to_string(numEpochs) + "epochs.save";

    FILE *out = fopen(fname.c_str(), "wb");
    fwrite(&numEpochs, sizeof(int), 1, out);
}

//calculate dev_u(t) = sign(t-tu)*|t-tu|^0.4 and save the result for saving the time
float TimeSVDPP::calcDev(int user, int timeArg) {
    if(Dev[user].count(timeArg) != 0) {
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
    clock_t time0 = clock();
    float preRmse = 1000;
    srand(time(NULL));
    int user, item, date, rating;
    float curRmse;
    for (size_t i = 0; i < 1000; ++i) {
        sgd();
        curRmse = cValidate(AVG);
        cout << "test_Rmse in step " << i << ": " << curRmse << endl;
        if(curRmse < preRmse - 0.00005) {
            preRmse = curRmse;
        }
        else{
            break;
        }
    }
    debugPrint("Outputting...\n");
    clock_t time1 = clock();
    FILE *fp = fopen(testFile.c_str(),"r");
    ofstream fout(outFile.c_str());
    while (fscanf(fp,"%d %d %d %d",&user, &item, &date, &rating) != EOF) {
        fout << predictScore(AVG, user - 1, item - 1, date - 1) << endl;
    }
    fclose(fp);
    fout.close();
    clock_t time2 = clock();
    float ms1 = diffclock(time2, time1);
    float ms2 = diffclock(time2, time0);
    printf("Outputting took %f ms\n", ms1);
    printf("Total took %f ms\n", ms2);
    cout << "final RMSE: " << curRmse << endl;
}


//function for cross validation
float TimeSVDPP::cValidate(float avg) {
    debugPrint("Cross validating...\n");
    clock_t time0 = clock();
    int userId,itemId,rating,t;
    int n = 0;
    float rmse = 0;
    for (const auto &ch:test_data) {
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
float TimeSVDPP::predictScore(float avg,int userId, int itemId,int time) {
    float tmp = 0.0;
    int sz = rowIndex[userId + 1] - rowIndex[userId];
    float sqrtNum = 0;
    if (sz > 1) {
        sqrtNum = 1. / (sqrt(sz));
    }
    for (size_t i = 0; i < factor; ++i) {
        tmp += (Pu[userId * factor + i] +sumMW[userId * factor + i]*sqrtNum) * Qi[itemId * factor + i];
    }
    float score = avg + Bu[userId] + Bi[itemId] + Bi_Bin[itemId * binNum + calcBin(time)] + Alpha_u[userId]*calcDev(userId,time) + Bu_t[userId][time] + tmp;

    if(score > 5) {
        score = 5;
    }
    if(score < 1) {
        score = 1;
    }
    return score;
}

//function for training
//update using stochastic gradient descent

void TimeSVDPP::sgd() {
    debugPrint("Updating using sgd...\n");
    clock_t time0 = clock();
    int userId,itemId,rating,time;
    for (userId = 0; userId < N_USERS; ++userId) {
        int userEnd = rowIndex[userId + 1];
        int userStart = rowIndex[userId];
        int sz = userEnd - userStart;
        float sqrtNum = 0;
        vector <float> tmpSum(factor, 0);
        if (sz > 1) {
            sqrtNum = 1/(sqrt(sz));
        }
        for (int k = 0; k < factor; ++k) {
            float sumy = 0;
            for (int colIdx = userStart; colIdx < userEnd; ++colIdx) {
                int itemI = columns[colIdx];
                sumy += y[itemI * factor + k];
            }
            sumMW[userId * factor + k] = sumy;
        }
        for (int colIdx = userStart; colIdx < userEnd; ++colIdx) {
            itemId = columns[colIdx];
            rating = values[colIdx];
            time = dates[colIdx];
            float prediction = predictScore(AVG, userId, itemId, time);
            float error = rating - prediction;
            Bu[userId] += G * (error - L * Bu[userId]);
            Bi[itemId] += G * (error - L * Bi[itemId]);
            Bi_Bin[itemId * binNum + calcBin(time)] += G * (error - L * Bi_Bin[itemId * binNum + calcBin(time)]);
            Alpha_u[userId] += G_alpha * (error * calcDev(userId,time)  - L_alpha * Alpha_u[userId]);
            Bu_t[userId][time] += G * (error - L * Bu_t[userId][time]);

            for (size_t k = 0; k < factor; ++k) {
                float uf = Pu[userId * factor + k];
                float mf = Qi[itemId * factor + k];
                Pu[userId * factor + k] += G * (error * mf - L_pq * uf);
                Qi[itemId * factor + k] += G * (error * (uf+sqrtNum*sumMW[userId * factor + k]) - L_pq * mf);
                tmpSum[k] += error*sqrtNum*mf;
            }
        }
        for (int colIdx = userStart; colIdx < userEnd; ++colIdx) {
            itemId = columns[colIdx];
            for (int k = 0; k < factor; ++k) {
                float tmpMW = y[itemId * factor + k];
                y[itemId * factor + k] += G*(tmpSum[k]- L_pq *tmpMW);
                sumMW[userId * factor + k] += y[itemId * factor + k] - tmpMW;
            }
        }
    }
    clock_t time1 = clock();
    float ms1 = diffclock(time1, time0);
    printf("First half SGD took %f ms\n", ms1);

    for (userId = 0; userId < N_USERS; ++userId) {
        int userEnd = rowIndex[userId + 1];
        int userStart = rowIndex[userId];
        int sz = userEnd - userStart;
        float sqrtNum = 0;
        if (sz > 1) {
            sqrtNum = 1.0/sqrt(sz);
        }
        for (int k = 0; k < factor; ++k) {
            float sumy = 0;
            for (int colIdx = userStart; colIdx < userEnd; ++colIdx) {
                int itemI = columns[colIdx];
                sumy += y[itemI * factor + k];
            }
            sumMW[userId * factor + k] = sumy;
        }
    }
    clock_t time2 = clock();
    float ms2 = diffclock(time2, time1);
    printf("Second half SGD took %f ms\n", ms2);
    G *= Decay;
    G_alpha *= Decay;
    clock_t time3 = clock();
    float ms3 = diffclock(time3, time0);
    printf("SGD took %f ms\n", ms3);
}
