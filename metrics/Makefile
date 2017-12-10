#You have to put pkg-config --cflags --libs opencv at the end of your g++ line. For example :
#g++ test.cpp -o test `pkg-config --cflags --libs opencv`
#Compiles
#g++ `pkg-config --cflags --libs opencv` test.cpp -o test

all:
	clang++ -std=c++11 host.cc -o host `pkg-config --cflags --libs opencv`
