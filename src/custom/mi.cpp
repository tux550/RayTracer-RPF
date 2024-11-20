// custom/mi.cpp*
#include "custom/mi.h"

// Function to compute the histogram of a data vector
vector<int> computeHistogram(const vector<double>& data, int bins, double minVal, double maxVal) {
    vector<int> hist(bins, 0);
    for (double value : data) {
        int bin = static_cast<int>((value - minVal) / (maxVal - minVal) * bins);
        bin = std::min(bin, bins - 1); // to avoid out-of-bounds indexing
        hist[bin]++;
    }
    return hist;
}

// Function to compute the joint histogram of two data vectors
vector<vector<int>> computeJointHistogram(const vector<double>& xData, const vector<double>& yData, int binsX, int binsY, double minX, double maxX, double minY, double maxY) {
    vector<vector<int>> jointHist(binsX, vector<int>(binsY, 0));
    
    for (size_t i = 0; i < xData.size(); ++i) {
        int binX = static_cast<int>((xData[i] - minX) / (maxX - minX) * binsX);
        binX = std::min(binX, binsX - 1);  // to avoid out-of-bounds indexing
        
        int binY = static_cast<int>((yData[i] - minY) / (maxY - minY) * binsY);
        binY = std::min(binY, binsY - 1);  // to avoid out-of-bounds indexing
        
        jointHist[binX][binY]++;
    }
    return jointHist;
}

// Function to compute the mutual information between two continuous variables X and Y
double MutualInformation(const vector<double>& xData, const vector<double>& yData, int binsX, int binsY) {
    // Step 1: Determine the range of the data
    double minX = *min_element(xData.begin(), xData.end());
    double maxX = *max_element(xData.begin(), xData.end());
    double minY = *min_element(yData.begin(), yData.end());
    double maxY = *max_element(yData.begin(), yData.end());

    // Step 2: Set default bin values (sqrt(N)) if bins are not provided
    if (binsX == -1) {
        binsX = static_cast<int>(sqrt(xData.size()));  // Default to sqrt(N) bins for X
    }
    if (binsY == -1) {
        binsY = static_cast<int>(sqrt(yData.size()));  // Default to sqrt(N) bins for Y
    }

    // Step 3: Compute histograms
    vector<int> histX = computeHistogram(xData, binsX, minX, maxX);
    vector<int> histY = computeHistogram(yData, binsY, minY, maxY);
    vector<vector<int>> jointHist = computeJointHistogram(xData, yData, binsX, binsY, minX, maxX, minY, maxY);

    // Step 4: Compute total number of samples
    double totalSamples = xData.size();

    // Step 5: Calculate marginal probabilities
    vector<double> probX(binsX), probY(binsY);
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
            if (pXY > 0) {
                mi += pXY * log(pXY / (probX[i] * probY[j]));
            }
        }
    }

    return mi;
}
