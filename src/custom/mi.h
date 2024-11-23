#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

void histcounts2(const std::vector<double>& x, const std::vector<double>& y, int nBinsX, int nBinsY, std::vector<std::vector<double>>& hist);
void marginalize(const std::vector<std::vector<double>>& hist, int nBinsX, int nBinsY, std::vector<double>& px, std::vector<double>& py);
int calculateBins(int n);

std::vector<int> computeHistogram(const std::vector<double>& data, int bins,
                                  double minVal, double maxVal);

std::vector<std::vector<int>> computeJointHistogram(
    const std::vector<double>& xData, const std::vector<double>& yData,
    int binsX, int binsY, double minX, double maxX, double minY, double maxY);

double MutualInformation(const std::vector<double>& xData,
                         const std::vector<double>& yData, int binsX = -1,
                         int binsY = -1);

bool buildHistograms(const std::vector<double>& dataX,
                    const std::vector<double>& dataY,
                    std::vector<int>& histX,
                    std::vector<int>& histY,
                    std::vector<std::vector<int>>& jointHist);