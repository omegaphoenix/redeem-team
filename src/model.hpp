#ifndef MODEL_HPP
#define MODEL_HPP
#include <string>

#define N_MOVIES 17770
#define N_USERS 480189

class Model {
    public:
        Model() {};
        virtual ~Model();
        void loadFresh(std::string fname);
        void loadSaved(std::string fname);
        virtual void train() = 0;
    private:
        virtual void generateMissing();
        float **ratings;
};

#endif // MODEL_HPP
