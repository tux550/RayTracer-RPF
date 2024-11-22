#include "custom/sd.h"

namespace pbrt {

SampleFMatrix XtoFMatrix(const SampleXMatrix &XMatrix) {
    SampleFMatrix FMatrix(XMatrix.size(),
                          SampleFVector(XMatrix[0].size(), SampleF()));
    for (size_t i = 0; i < XMatrix.size(); ++i) {
        for (size_t j = 0; j < XMatrix[0].size(); ++j) {
            SampleF F = SampleData::getFeatures(XMatrix[i][j]);
            FMatrix[i][j] = F;
        }
    }
    return FMatrix;
}

SampleDataSetMatrix normalizedSamples(const SampleDataSetMatrix &samples,
                                      const SampleXMatrix &meanMatrix,
                                      const SampleXMatrix &stdDevMatrix) {
    SampleDataSetMatrix normalizedSamples = SampleDataSetMatrix(
        samples.size(),
        SampleDataSetVector(samples[0].size(), SampleDataSet()));
    for (size_t i = 0; i < samples.size(); ++i) {
        for (size_t j = 0; j < samples[0].size(); ++j) {
            SampleDataSet dataset;
            for (const SampleData &sf : samples[i][j]) {
                dataset.push_back(
                    sf.normalized(meanMatrix[i][j], stdDevMatrix[i][j]));
            }
            normalizedSamples[i][j] = dataset;
        }
    }
    return normalizedSamples;
}

}  // namespace pbrt
