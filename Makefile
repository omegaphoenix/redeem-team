CXX = g++
CXXFLAGS = -std=c++11 -Wall -g
PROG = naive_svd
NAIVE_SVD_FILES = $(addprefix src/, naive_svd.cpp model.cpp)

all: init naive_svd

init:
	mkdir -p bin out model
	if [ ! -f "data/um/5-1.dta" ]; \
then \
	sed 's/0$$/1/' data/um/5.dta > data/um/5-1.dta; \
fi

naive_svd: $(NAIVE_SVD_FILES:.cpp=.o)
	$(CXX) $(CFLAGS) -o bin/$@ $^

clean:
	rm -f bin/* src/*.o

.PHONY: all clean init
