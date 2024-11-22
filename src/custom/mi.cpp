#include "custom/mi.h"

#include <vector>

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
            binX = std::max(binX, 0);
        }
        int binY = 0;
        if (maxY != minY) {
            binY = static_cast<int>((yData[i] - minY) / (maxY - minY) * binsY);
            binY =
                std::min(binY, binsY - 1);  // to avoid out-of-bounds indexing
            binY = std::max(binY, 0);
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

    // Step 2: Set default bin values (sqrt(N)) if bins are not provided
    if (binsX == -1) {
        binsX = std::max(
            1, static_cast<int>(
                   sqrt(xData.size())));  // Default to sqrt(N) bins for X
    }
    if (binsY == -1) {
        binsY = std::max(
            1, static_cast<int>(
                   sqrt(yData.size())));  // Default to sqrt(N) bins for Y
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
        probX[i] = histX[i] / totalSamples;
    }
    for (int j = 0; j < binsY; ++j) {
        probY[j] = histY[j] / totalSamples;
    }

    // Step 6: Calculate joint probabilities and mutual information
    double mi = 0.0;
    for (int i = 0; i < binsX; ++i) {
        for (int j = 0; j < binsY; ++j) {
            double pXY = jointHist[i][j] / totalSamples;
            double probXtimesY = probX[i] * probY[j];  // Division By Zero check
            if (pXY > 0 && probXtimesY != 0) {
                mi += pXY * log(pXY / probXtimesY);
            }
        }
    }

    return mi;
}
