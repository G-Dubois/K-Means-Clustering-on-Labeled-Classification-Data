all:	kmeans

kmeans:	kmeans.cpp
	g++ -std=c++14 kmeans.cpp -o kmeans
