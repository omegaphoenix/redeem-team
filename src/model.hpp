#ifndef MODEL_HPP
#define MODEL_HPP
#include <string>
#include <time.h>
#include <vector>

#define N_MOVIES 17770
#define N_USERS 458293

class Model {
    public:
        Model();
        std::vector<std::vector<char> > ratings;
        virtual ~Model();
        void loadFresh(std::string fname);
        void loadSaved(std::string fname);
        virtual void train() = 0;
    private:
        virtual void generateMissing();
};

#endif // MODEL_HPP
