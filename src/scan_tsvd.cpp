#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include "time_svd_pp.hpp"

using namespace std;

vector<int> binsize = {30, 40};
vector<int> factors = {150, 160, 170, 180, 190};

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

int main() {
    string trainFile = "1.dta";
    string crossFile = "data/um/4.dta";
    string testFile = "data/um/5-1.dta";

    vector<Node> nodes;
    for (int i = 0; i < binsize.size(); ++i) {
        int binNum = binsize[i];
        for (int j = 0; j < factors.size(); ++j) {
            int factor = factors[j];

            bool fileExists = false;
            string fname;
            for (int epoch = 999; epoch > 0; --epoch) {
                fname = "model/timesvdpp/" + std::to_string(factor)
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
    return 0;
}
