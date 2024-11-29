#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_RANDOM_PARAMETER_FILTERING_H
#define PBRT_RANDOM_PARAMETER_FILTERING_H

// custom/rpf.h*
#include <vector>

#include "camera.h"
#include "custom/ops.h"
#include "custom/sample_film.h"
#include "custom/sd.h"
#include "integrator.h"
#include "lightdistrib.h"
#include "pbrt.h"
#include "scene.h"
#include "visualization/vis.h"
#include "custom/mi.h"


namespace pbrt {
// Visualization
void visualizeSD(const SampleDataSetMatrix &sdMat, const std::string &filename);


struct ComputeMIBuffer {
    // Reserve buffer for mutual information calc
    std::vector<int> bufferQuantizedX;
    std::array<double, MAX_NUM_OF_BINS> bufferHistX;
    std::vector<int> bufferQuantizedY;
    std::array<double, MAX_NUM_OF_BINS> bufferHistY;
    std::array<std::array<double, MAX_NUM_OF_BINS>, MAX_NUM_OF_BINS> bufferJointHist;

    // Constructor
    ComputeMIBuffer(int neigborhood_size) {
        bufferQuantizedX = std::vector<int>(neigborhood_size);
        bufferQuantizedY = std::vector<int>(neigborhood_size);
    }
};


// RPFIntegrator
class RPFIntegrator : public Integrator {
  public:
    // RPFIntegrator Public Methods
    RPFIntegrator(int maxDepth, std::shared_ptr<const Camera> camera,
                  std::shared_ptr<Sampler> sampler, const Bounds2i &pixelBounds,
                  Float rrThreshold = 1,
                  const std::string &lightSampleStrategy = "spatial");
    void Preprocess(const Scene &scene, Sampler &sampler);
    void Render(const Scene &scene);
    void Li(const RayDifferential &ray, const Scene &scene, Sampler &sampler,
            MemoryArena &arena, SampleData &sf, int depth = 0) const;

  protected:
    // RPFIntegrator Protected Data
    std::shared_ptr<const Camera> camera;

  private:
    // RPFIntegrator Private Data
    const int maxDepth;
    const Float rrThreshold;
    const std::string lightSampleStrategy;
    std::unique_ptr<LightDistribution> lightDistribution;

    std::shared_ptr<Sampler> sampler;
    const Bounds2i pixelBounds;
    // Utility functions
    void getXStatsPerPixel(const SampleDataSetMatrix &samples,
                           SampleXMatrix &meanMatrix,
                           SampleXMatrix &stdDevMatrix);

    SampleDataSetMatrix getNeighborhoodSamples(
        const SampleDataSetMatrix &samples, const SampleFMatrix &meanMatrix,
        const SampleFMatrix &stdDevMatrix,
        size_t box_size  // Always odd
    );

    void FillSampleFilm(SamplingFilm &samplingFilm, const Scene &scene,
                        const int tileSize);
    void FillMeanAndStddev(const SamplingFilm &samplingFilm,
                           SampleFMatrix &pixelFmeanMatrix,
                           SampleFMatrix &pixelFstdDevMatrix,
                           const int tileSize);

    void ComputeCFWeights(const SampleDataSet &neighborhood, SampleC &Alpha_k,
                          SampleF &Beta_k, int t, double &W_r_c, ComputeMIBuffer &buffer);
    void ApplyRPFFilter(SamplingFilm &samplingFilm, const int tileSize,
                        int box_size, int t  //= 3;
    );
};

RPFIntegrator *CreateRPFIntegrator(const ParamSet &params,
                                   std::shared_ptr<Sampler> sampler,
                                   std::shared_ptr<const Camera> camera);

}  // namespace pbrt

#endif  // PBRT_INTEGRATORS_RPF_PATH_H
