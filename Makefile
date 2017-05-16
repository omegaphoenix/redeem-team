CXX = g++
CXXFLAGS = -std=c++11 -Wall -g -Wshadow -O3
PROG = naive_svd
RBM_PROG = rbm
NAIVE_SVD_FILES = $(addprefix src/, naive_svd_main.cpp naive_svd.cpp model.cpp)
NAIVE_SVD_CV_FILES = $(addprefix src/, validate_naive_svd.cpp naive_svd.cpp model.cpp)
BASELINE_FILES = $(addprefix src/, baseline.cpp baseline_main.cpp model.cpp)
SVD_PLUS_FILES = $(addprefix src/, svd_plusplus_main.cpp svd_plusplus.cpp baseline.cpp model.cpp)
SVD_PLUS_CV_FILES = $(addprefix src/, validate_svd_plusplus.cpp svd_plusplus.cpp baseline.cpp model.cpp)
RBM_FILES = $(addprefix src/, rbm.cpp model.cpp)
# MODEL_FILES = $(addprefix src/, model.cpp)
# BASELINE_FILES = $(addprefix src/, baseline.cpp model.cpp)


all: init naive_svd validate_naive_svd

init:
	mkdir -p bin log out model out/rbm model/naive_svd model/rbm
	if [ ! -f "data/um/5-1.dta" ]; \
then \
	sed 's/0$$/1/' data/um/5.dta > data/um/5-1.dta; \
fi

# baseline: $(BASELINE_FILES:.cpp=.o)
	# $(CXX) $(CFLAGS) -o bin/$@ $^

# model: $(MODEL_FILES:.cpp=.o)
	# $(CXX) $(CFLAGS) -o bin/$@ $^

naive_svd: $(NAIVE_SVD_FILES:.cpp=.o)
	$(CXX) $(CFLAGS) -o bin/$@ $^

svd_plus: $(SVD_PLUS_FILES:.cpp=.o)
	$(CXX) -O3 $(CFLAGS) -o bin/$@ $^

validate_naive_svd: $(NAIVE_SVD_CV_FILES:.cpp=.o)
	$(CXX) $(CFLAGS) -o bin/$@ $^

validate_svd_plus: $(SVD_PLUS_CV_FILES:.cpp=.o)
	$(CXX) -O3 $(CFLAGS) -o bin/$@ $^

run_nsvd: validate_naive_svd
	bin/validate_naive_svd 2> log/validate_nsvd.log

rbm: $(RBM_FILES:.cpp=.o)
	$(CXX) $(CFLAGS) -o bin/rbm $^

clean:
	rm -f -R bin/* src/*.o

.PHONY: all clean init
