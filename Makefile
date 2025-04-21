test: main.cpp
	g++ -msse2 -Wall -march=native *.cpp *.hpp -o test
