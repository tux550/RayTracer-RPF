#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_INTEGRATORS_RPF_PATH_H
#define PBRT_INTEGRATORS_RPF_PATH_H

// integrators/rpf.h*
#include <ImfRgbaFile.h>
#include <ImfArray.h>
#include <vector>
#include "pbrt.h"
#include "integrator.h"
#include "camera.h"
#include "scene.h"
#include "lightdistrib.h"



namespace pbrt {

struct BasicRGB {
  double r;
  double g;
  double b;
  BasicRGB(double r, double g, double b): r(r), g(g), b(b) {};
};

struct FeatureVector {
  // FIRST INTERSECTION
  Normal3f n0; // Normal
  Point3f p0;  // World-space position
              // Texture?
  // SECOND INTERSECTION
  Normal3f n1; // Normal
  Point3f p1;  // World-space position

  // CONSTRUCTOR
  FeatureVector():
    n0(Normal3f(0, 0, 0)),
    p0(Point3f(0, 0, 0)),
    n1(Normal3f(0, 0, 0)),
    p1(Point3f(0, 0, 0))
  {};
};

void writeFVMat(const std::vector<std::vector<
 std::vector<FeatureVector>>> &fvMat, const std::string &filename);

struct SampleData {
  // SampleData Public Data
  Point2f pFilm;
  Point2f pLens;
  Spectrum L;
  Float rayWeight;
  FeatureVector fv;

  // CONSTRUCTOR
  SampleData(const Point2f &pFilm, const Spectrum &L, Float rayWeight, const FeatureVector &fv):
    pFilm(pFilm),
    L(L),
    rayWeight(rayWeight),
    fv(fv)
  {};
};

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
    Spectrum Li(
      const RayDifferential &ray,
      const Scene &scene,
      Sampler &sampler,
      MemoryArena &arena,
      FeatureVector &fv,
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
};

RPFIntegrator *CreateRPFIntegrator(
  const ParamSet &params,
  std::shared_ptr<Sampler> sampler,
  std::shared_ptr<const Camera> camera
); 

}



#endif  // PBRT_INTEGRATORS_RPF_PATH_H
