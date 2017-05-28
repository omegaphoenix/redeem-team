#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include "time_svd_pp.hpp"

using namespace std;

vector<int> binsize = {30};
vector<int> factors = {240, 270, 700, 900};

const float AVG = 3.60073;     // average score

class Node {
public:
    string modelName;
    float score;
    bool operator< (const Node& other) {
        return score < other.score;
    }
};

bool compare(Node a, Node b) {
    return a.score < b.score;
}

void run(string trainFile, string crossFile, string testFile,
         string dir) {
    vector<Node> nodes;
    for (int i = 0; i < binsize.size(); ++i) {
        int binNum = binsize[i];
        for (int j = 0; j < factors.size(); ++j) {
            int factor = factors[j];

            bool fileExists = false;
            string fname;
            for (int epoch = 999; epoch > 0; --epoch) {
                fname = dir + trainFile + "-trained_"
                        + std::to_string(factor)
                        + "factors_" + std::to_string(binNum) + "bins_"
                        + std::to_string(epoch) + "epochs.save";
                std::ifstream f(fname.c_str());
                if (f.good()) {
                    fileExists = true;
                    break;
                }
            }

            TimeSVDPP* cur;
            if (fileExists) {
                cur = loadTSVDpp(fname, trainFile, crossFile, testFile);
            }
            else {
                cur = new TimeSVDPP(false,0,binNum,factor,NULL,NULL,NULL,NULL,
                              NULL,NULL,NULL,NULL,NULL,NULL,
                              trainFile, crossFile, testFile);
            }
            cout << "About to train " << fname << "\n";
            cur->train("");
            Node n;
            n.modelName = fname;
            n.score = cur->cValidate(AVG);
            nodes.push_back(n);
            delete cur;
        }
    }
    sort(nodes.begin(), nodes.end(), compare);

    for (int i = 0; i < nodes.size(); i++) {
        Node n = nodes[i];
        cerr << n.modelName << ": " << n.score << "\n";
    }
    cerr << "\n";
}

int main() {
    string trainFile = "1.dta";
    string crossFile = "data/um/4.dta";
    string testFile = "data/um/5-1.dta";
    string dir = "model/timesvd/";
    testFile = "data/um/5-1.dta"; // For nn-blending
    // dir = "/Volumes/dqu/cs156/timesvdpp/models/30bin/";

    // nn Training data
    run("1.dta", crossFile, "data/um/23.dta", dir);
    // nn Validation data
    run("1.dta", crossFile, "data/um/4.dta", dir);
    // Data to actually blend
    run("1.dta", crossFile, "data/um/5-1.dta", dir);
    return 0;
}
