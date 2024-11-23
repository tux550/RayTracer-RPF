#include "custom/mi.h"

#include <vector>

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
/*
// Function to compute the histogram of a data vector
bool buildHistograms(const std::vector<double>& dataX,
                    const std::vector<double>& dataY,
                    std::vector<int>& histX,
                    std::vector<int>& histY,
                    std::vector<std::vector<int>>& jointHist) {
   
    // Get min & max values
    double minX = *std::min_element(dataX.begin(), dataX.end());
    double maxX = *std::max_element(dataX.begin(), dataX.end());
    double minY = *std::min_element(dataY.begin(), dataY.end());
    double maxY = *std::max_element(dataY.begin(), dataY.end());

    // Get number of bins round(max-min)
    int binsX = std::round(maxX - minX) +1;
    int binsY = std::round(maxY - minY) +1;
    
    // If only 1 bin, return false
    //if (binsX == 1 || binsY == 1) {
    //    return false;
    //}

    // Init histograms
    histX.resize(binsX, 0);
    histY.resize(binsY, 0);
    jointHist.resize(binsX, std::vector<int>(binsY, 0));

    // Fill histograms
    for (size_t i = 0; i < dataX.size(); ++i) {
        // calc bin as round(value - min)
        int binX = std::round(dataX[i] - minX);
        int binY = std::round(dataY[i] - minY);
        histX[binX]++;
        histY[binY]++;
        jointHist[binX][binY]++;
    }
    return true;
}

// Function to compute the mutual information between two continuous variables X
// and Y
double MutualInformation(const std::vector<double>& xData,
                         const std::vector<double>& yData, int binsX,
                         int binsY) {
    // Build histograms
    std::vector<std::vector<int>> jointHist;
    std::vector<int> histX;
    std::vector<int> histY;
    if (!buildHistograms(xData, yData, histX, histY, jointHist)) {
        return 0.0; // If returns false, then one of the variables is "constants". Then, MI is 0
    }

    // Compute total number of samples
    double totalSamples = xData.size();

    // Step 6: Calculate joint probabilities and mutual information
    double mi = 0.0;
    for (int i = 0; i < binsX; ++i) {
        double pX = ( (double) histX[i] ) / totalSamples;
        for (int j = 0; j < binsY; ++j) {
            double pY = ( (double) histY[j] ) / totalSamples;
            double pXY = ( (double) jointHist[i][j] ) / totalSamples;
            double probXtimesY = pX * pY;  // Division By Zero check
            if (pXY > 0 && probXtimesY != 0) {
                mi += pXY * log(pXY / probXtimesY);
            }
        }
    }

    return mi;
}
*/


// Function to compute the histogram of a data vector
std::vector<int> computeHistogram(const std::vector<double>& data, int bins,
                                  double minVal, double maxVal) {
    std::vector<int> hist(bins, 0);
    if (maxVal == minVal) {
        // If all data points are identical, dump all in first
        hist[0] = data.size();
        return hist;  // Return histogram with all zeros
    }

    for (double value : data) {
        int bin = static_cast<int>((value - minVal) / (maxVal - minVal) * bins);
        bin = std::min(bin, bins - 1);  // to avoid out-of-bounds indexing
        bin = std::max(bin, 0);
        hist[bin]++;
    }
    return hist;
}

// Function to compute the joint histogram of two data vectors
std::vector<std::vector<int>> computeJointHistogram(
    const std::vector<double>& xData, const std::vector<double>& yData,
    int binsX, int binsY, double minX, double maxX, double minY, double maxY) {
    std::vector<std::vector<int>> jointHist(binsX, std::vector<int>(binsY, 0));

    for (size_t i = 0; i < xData.size(); ++i) {
        int binX = 0;
        if (maxX != minX) {
            binX = static_cast<int>((xData[i] - minX) / (maxX - minX) * binsX);
            binX =
                std::min(binX, binsX - 1);  // to avoid out-of-bounds indexing
        }
        int binY = 0;
        if (maxY != minY) {
            binY = static_cast<int>((yData[i] - minY) / (maxY - minY) * binsY);
            binY =
                std::min(binY, binsY - 1);  // to avoid out-of-bounds indexing
        }
        jointHist[binX][binY]++;
    }
    return jointHist;
}

// Function to compute the mutual information between two continuous variables X
// and Y
double MutualInformation(const std::vector<double>& xData,
                         const std::vector<double>& yData, int binsX,
                         int binsY) {
    // Step 1: Determine the range of the data
    double minX = *min_element(xData.begin(), xData.end());
    double maxX = *max_element(xData.begin(), xData.end());
    double minY = *min_element(yData.begin(), yData.end());
    double maxY = *max_element(yData.begin(), yData.end());

    if (minX == maxX || minY == maxY) {
        return 0.0;  // If all data points are identical, return 0
    }
    //binsX = 2;
    //binsY = 2;

    // Step 2: Set default bin values = log2(N)+1
    if (binsX == -1) {
        binsX = std::log2(xData.size()) + 1;
    }
    if (binsY == -1) {
        binsY = std::log2(yData.size()) + 1;
    }

    // Step 3: Compute histograms
    std::vector<int> histX = computeHistogram(xData, binsX, minX, maxX);
    std::vector<int> histY = computeHistogram(yData, binsY, minY, maxY);
    std::vector<std::vector<int>> jointHist = computeJointHistogram(
        xData, yData, binsX, binsY, minX, maxX, minY, maxY);

    // Step 4: Compute total number of samples
    double totalSamples = xData.size();

    // Step 5: Calculate marginal probabilities
    std::vector<double> probX(binsX), probY(binsY);
    for (int i = 0; i < binsX; ++i) {
        probX[i] = ( (double) histX[i] ) / totalSamples;
    }
    for (int j = 0; j < binsY; ++j) {
        probY[j] = ( (double) histY[j] ) / totalSamples;
    }

    // Step 6: Calculate joint probabilities and mutual information
    double mi = 0.0;
    for (int i = 0; i < binsX; ++i) {
        for (int j = 0; j < binsY; ++j) {
            double pXY = ( (double) jointHist[i][j] ) / totalSamples;
            double probXtimesY = probX[i] * probY[j];  // Division By Zero check
            if (pXY > 0 && probXtimesY != 0) {
                mi += pXY * log(pXY / probXtimesY);
            }
        }
    }

    return mi;
}
