#include <iostream>
#include <numeric>
#include <vector>
#include <time.h>
#include "naive_svd.hpp"

using namespace std;

int main(int argc, char **argv) {
    // Speed up stdio operations
    std::ios_base::sync_with_stdio(false);

    NaiveSVD* nsvd = new NaiveSVD();

    // Load in COO format into ratings vector
    nsvd->load("all.dta");
    printf("numRatings all %d\n", nsvd->numRatings);

    vector<unsigned int>* userCount = new vector<unsigned int>(N_USERS);
    for (int i = 0; i < N_USERS; ++i) {
        (*userCount)[i] = nsvd->rowIndex[i + 1] - nsvd->rowIndex[i];
    }

    // vector<int>* ind1 = sort_indexes(movieCount);
    int min = 10;
    int sum = 0;
    for (int i = 0; i < N_USERS; ++i) {
        unsigned int count = (*userCount)[i];
        if (count < min) {
            cout << "user " << i << " count " << count << endl;
            sum += count;
        }
    }
    cout << "sum " << sum << endl;
    /*
    for (int i = 0; i <= N_USERS; ++i) {
        unsigned int mov = (*userCount)[i];
        cout << i << " " << mov << endl;
    }
    */

	delete nsvd;
	delete userCount;
    return 0;
}
