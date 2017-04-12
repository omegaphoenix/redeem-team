CXX = g++
CXXFLAGS = -std=c++11 -Wall -g
PROG = naive_svd
NAIVE_SVD_FILES = $(addprefix src/, naive_svd.cpp model.cpp)

all: init naive_svd

init:
	mkdir -p bin

naive_svd: $(NAIVE_SVD_FILES:.cpp=.o)
	$(CXX) $(CFLAGS) -o bin/$@ $^

clean:
	rm src/*.o

.PHONY: all clean init