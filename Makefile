CXX = g++
CXXFLAGS = -std=c++11 -Wall -g
PROG = naive_svd
NAIVE_SVD_FILES = $(addprefix src/, naive_svd.cpp model.cpp)
KNN_FILES = $(addprefix src/, knn.cpp model.cpp)

all: init naive_svd knn

init:
	mkdir -p bin

naive_svd: $(NAIVE_SVD_FILES:.cpp=.o)
	$(CXX) $(CFLAGS) -o bin/$@ $^

knn: $(KNN_FILES:.cpp=.o)
	$(CXX) $(CFLAGS) -o bin/$@ $^

clean:
	rm src/*.o

.PHONY: all clean init
