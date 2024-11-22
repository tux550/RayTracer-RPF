#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

vector<int> computeHistogram(const vector<double>& data, int bins,
                             double minVal, double maxVal);

vector<vector<int>> computeJointHistogram(const vector<double>& xData,
                                          const vector<double>& yData,
                                          int binsX, int binsY, double minX,
                                          double maxX, double minY,
                                          double maxY);

double MutualInformation(const vector<double>& xData,
                         const vector<double>& yData, int binsX = -1,
                         int binsY = -1);
