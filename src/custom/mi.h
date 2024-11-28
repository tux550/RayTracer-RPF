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


double MutualInformation(std::vector<double> const& x_vec,
                         std::vector<double> const& x_prob, double x_min,
                         double x_max, const std::vector<double>& y_vec,
                         std::vector<double> const& y_prob, double y_min,
                         double y_max);
