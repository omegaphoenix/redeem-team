CXX = g++
CXXFLAGS = -std=c++11 -Wall -g
PROG = naive_svd
NAIVE_SVD_FILES = $(addprefix src/, naive_svd_main.cpp naive_svd.cpp model.cpp)
NAIVE_SVD_CV_FILES = $(addprefix src/, validate_naive_svd.cpp naive_svd.cpp model.cpp)


all: init naive_svd validate_naive_svd

init:
	mkdir -p bin log out model model/naive_svd
	if [ ! -f "data/um/5-1.dta" ]; \
then \
	sed 's/0$$/1/' data/um/5.dta > data/um/5-1.dta; \
fi

naive_svd: $(NAIVE_SVD_FILES:.cpp=.o)
	$(CXX) $(CFLAGS) -o bin/$@ $^

validate_naive_svd: $(NAIVE_SVD_CV_FILES:.cpp=.o)
	$(CXX) $(CFLAGS) -o bin/$@ $^

run_nsvd: validate_naive_svd
	bin/validate_naive_svd 2> log/validate_nsvd.log

clean:
	rm -f bin/* src/*.o

.PHONY: all clean init
