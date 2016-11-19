//  Project:		K-Means Clustering
//  Date:			11 - 17 - 2016
//  Class:			CSCI 4350 - Intro to Artificial Intelligence
//  Assignment:		Open Lab 4 - Unsupervised Learning
//  Filename:		K-Means Driver Code (kmeans.cpp)
//  Author:			Grayson M. Dubois
//	Instructor:		Joshua L. Phillips
//  Institution:	Middle Tennessee State University
//  Department:		Computer Science

#include <iostream>
#include <map>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

// Example stores the values of all features and classification
// of a single example in the data set
struct Example {
	vector<double> features;
	int classification;

	Example(int numberOfFeatures) {
		features.resize(numberOfFeatures);
	}
	
	void setFeatureSize(int numberOfFeatures) {
		features.resize(numberOfFeatures);
	}
};

typedef vector<Example>::iterator ExampleIterator;

// Function Declarations
double d(Example*, vector<double>*);
bool isSame(vector<double>&, vector<double>&);
ostream& operator<<(ostream& out, const vector<double> vec);
ostream& operator<<(ostream& out, const Example ex);
istream& operator>>(istream& in,  Example& ex);


int main(int argc, char* argv[]) {

	// Configuration
	int seed, numberOfClusters, numberOfFeatures;
	string trainingFilename, testingFilename;

	// Data
	vector<Example> trainingData, testingData;
	vector<vector<double>> centers;
	vector<vector<Example*>> clusters;

	// Catch error in number of command line arguments
	if (argc != 6) {
		cerr << "usage: kmeans [seed] [number of clusters] [number of features] [training file name] [testing file name]\n";
		return 9;
	} else {
		// Get settings from command line
		seed = atoi(argv[1]);
		srand(seed);
		
		numberOfClusters = atoi(argv[2]);
		clusters.resize(numberOfClusters);

		numberOfFeatures = atoi(argv[3]);
		
		trainingFilename = argv[4];
		testingFilename = argv[5];
	}

	// Read in the training data
	ifstream fin(trainingFilename);
	while (!fin.eof()) {
		Example ex(numberOfFeatures);
		fin >> ex;
		if (fin.eof()) break;
		trainingData.push_back(ex);
	}

	// Read in the testing data
	fin.close();
	fin.open(testingFilename);
	while (!fin.eof()) {
		Example ex(numberOfFeatures);
		fin >> ex;
		if (fin.eof()) break;
		testingData.push_back(ex);
	}
	
	// Choose K random centers from the training data
	for (int i = 0; i < numberOfClusters; i++) {
		Example& temp = trainingData[rand()%trainingData.size()];
		vector<double> featureVector(numberOfFeatures);
		for (int x = 0; x < numberOfFeatures; x++) {
			featureVector[x] = temp.features[x];
		}
		centers.push_back(featureVector);
	}

	bool centersMoved = false;

	do {
		centersMoved = false;
		clusters.clear();
		clusters.resize(numberOfClusters);

		// Cluster the examples based on distance
		for (auto& example : trainingData) {
			pair<int, double> shortestDistance(0, d(&example, &(centers[0])));
			for (int i = 1; i < numberOfClusters; i++) {
				double distance = d(&example, &(centers[i]));
				if (distance < shortestDistance.second) {
					shortestDistance = make_pair(i, distance);
				}
			}

			clusters[shortestDistance.first].push_back(&example);
		}

		// Find the new centers based on the current clusters
		for (int i = 0; i < clusters.size(); i++) {
			vector<double> newCenter(numberOfFeatures);
			for (int f = 0; f < numberOfFeatures; f++) {
				double sum = 0.0;
				for (int j = 0; j < clusters[i].size(); j++) {
					sum += clusters[i][j]->features[f];
				}
				newCenter[f] = sum / clusters[i].size();
			}

			if (!isSame(centers[i], newCenter)) {
				centersMoved = true;
				centers[i] = newCenter;
			}
		}

	} while (centersMoved);

	// Assign a classification to each cluster by majority vote
	int classifications[numberOfClusters];
	for (int i = 0; i < numberOfClusters; i++) {
		map<int, int> classMap;
		for (int j = 0; j < clusters[i].size(); j++) {
			classMap[clusters[i][j]->classification]++;
		}
		
		int maxClass = 0;
		for (auto classification : classMap) {
			if (classification.second > classMap[maxClass]) {
				maxClass = classification.first;
			}
		}
		classifications[i] = maxClass;
	}


	// Classify testing data and compare to actual classification labels
	int numberCorrectlyClassified = 0;
	for (auto& example : testingData) {
		pair<int, double> shortestDistance(0, d(&example, &(centers[0])));
		for (int i = 1; i < numberOfClusters; i++) {
			double distance = d(&example, &(centers[i]));
			if (distance < shortestDistance.second) {
				shortestDistance = make_pair(i, distance);
			}
		}

		if (classifications[shortestDistance.first] == example.classification) {
			numberCorrectlyClassified++;
		}
	}

	cout << numberCorrectlyClassified << "\n";
	
	return 0;
}

bool isSame(vector<double>& v1, vector<double>& v2) {

	if (v1.size() != v2.size()) {
		return false;
	}

	for (int i = 0; i < v1.size(); i++) {
		if ((float)v1[i] != (float)v2[i]) {
			return false;
		}
	}

	return true;
}

double d(Example* ex, vector<double>* center) {

	double sum = 0.0;

	for (int i = 0; i < (*center).size(); i++) {
		sum += pow((*ex).features[i] - (*center)[i], 2);
	}

	return sqrt(sum);
}

ostream& operator<<(ostream& out, const vector<double> vec) {
	if (vec.size() > 0) {
		cout << vec[0];
		for (int i = 1; i < vec.size(); i++) {
			cout << "\t" << vec[i];
		}
	}
	return out;
}

ostream& operator<<(ostream& out, const Example ex) {
	if (ex.features.size() > 0) {
		out << ex.features[0];
		for (int i = 1; i < ex.features.size(); i++) {
			out << "\t" << ex.features[i];
		}
		out << "\t" << ex.classification;
	}
	return out;
}

istream& operator>>(istream& in,  Example& ex) {
	for (auto& feature : ex.features) {
		in >> feature;
	}
	in >> ex.classification;
	return in;
}
