#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_RANDOM_PARAMETER_FILTERING_H
#define PBRT_RANDOM_PARAMETER_FILTERING_H

// custom/rpf.h*
#include <vector>
#include "visualization/vis.h"
#include "custom/sd.h"
#include "custom/ops.h"
#include "custom/sample_film.h"
#include "pbrt.h"
#include "integrator.h"
#include "camera.h"
#include "scene.h"
#include "lightdistrib.h"



namespace pbrt {
// Visualization
void visualizeSD(const SampleDataSetMatrix  &sdMat, const std::string &filename);

// RPFIntegrator
class RPFIntegrator : public Integrator {
  public:
    // RPFIntegrator Public Methods
    RPFIntegrator(
      int maxDepth,
      std::shared_ptr<const Camera> camera,
      std::shared_ptr<Sampler> sampler,
      const Bounds2i &pixelBounds,
      Float rrThreshold = 1,
      const std::string &lightSampleStrategy = "spatial"
    );
    void Preprocess(const Scene &scene, Sampler &sampler);
    void Render(const Scene &scene);
    void Li(
      const RayDifferential &ray,
      const Scene &scene,
      Sampler &sampler,
      MemoryArena &arena,
      SampleData &sf,
      int depth = 0
    ) const;
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
    void getXStatsPerPixel(
      const SampleDataSetMatrix &samples,
      SampleXMatrix &meanMatrix,
      SampleXMatrix &stdDevMatrix
    );  

    SampleDataSetMatrix getNeighborhoodSamples(
      const SampleDataSetMatrix &samples,
      const SampleFMatrix &meanMatrix,
      const SampleFMatrix &stdDevMatrix,
      size_t box_size // Always odd
    );


};

RPFIntegrator *CreateRPFIntegrator(
  const ParamSet &params,
  std::shared_ptr<Sampler> sampler,
  std::shared_ptr<const Camera> camera
); 

}



#endif  // PBRT_INTEGRATORS_RPF_PATH_H
