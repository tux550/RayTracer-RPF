#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include "pbrt.h"
#include "progressreporter.h"
#define MAX_NUM_OF_BINS int(10) 
std::vector<int> computeHistogram(const std::vector<double>& data, int bins,
                                  double minVal, double maxVal);

std::vector<std::vector<int>> computeJointHistogram(
    const std::vector<double>& xData, const std::vector<double>& yData,
    int binsX, int binsY, double minX, double maxX, double minY, double maxY);


double MutualInformation(
    const std::vector<double>& xData,
    const std::vector<double>& yData,
    int dataLength,
    std::vector<int>& bufferQuantizedX,
    std::vector<int>& bufferQuantizedY,
    std::array<double, MAX_NUM_OF_BINS>& bufferHistX,
    std::array<double, MAX_NUM_OF_BINS>& bufferHistY,
    std::array<std::array<double, MAX_NUM_OF_BINS>, MAX_NUM_OF_BINS>& bufferJointHist
); 

//double MutualInformation(const std::vector<double>& xData, const std::vector<double>& yData);
