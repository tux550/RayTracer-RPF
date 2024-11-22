#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

std::vector<int> computeHistogram(const std::vector<double>& data, int bins,
                                  double minVal, double maxVal);

std::vector<std::vector<int>> computeJointHistogram(
    const std::vector<double>& xData, const std::vector<double>& yData,
    int binsX, int binsY, double minX, double maxX, double minY, double maxY);

double MutualInformation(const std::vector<double>& xData,
                         const std::vector<double>& yData, int binsX = -1,
                         int binsY = -1);
