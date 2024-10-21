#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_VISUALIZATION_H
#define PBRT_VISUALIZATION_H

// custom/rpf.h*
#include <vector>
#include "visualization/vis.h"
#include "pbrt.h"
#include "integrator.h"
#include "camera.h"
#include "scene.h"
#include "lightdistrib.h"



namespace pbrt {

// SampleFeatures: n0, p0, n1, p1
typedef std::array<double, 12> SampleF;
typedef std::vector<SampleF> SampleFVector;
typedef std::vector<SampleFVector> SampleFMatrix;
// SamplePosition: pFilm
typedef std::array<double, 2> SampleP;
// SampleRandom: pLens
typedef std::array<double, 2> SampleR;

// Template for ArrayOperations
template <typename T, size_t N>
std::array<T,N> sumArrays(const std::array<T,N> &a, const std::array<T,N> &b) {
  std::array<T,N> result;
  for (size_t i = 0; i < N; ++i) {
    result[i] = a[i] + b[i];
  }
  return result;
}

template <typename T, size_t N>
std::array<T,N> divideArray(const std::array<T,N> &a, double scalar) {
  std::array<T,N> result;
  for (size_t i = 0; i < N; ++i) {
    result[i] = a[i] / scalar;
  }
  return result;
}

template <typename T, size_t N>
std::array<T,N> divideArrays(const std::array<T,N> &a, const std::array<T,N> &b) {
  std::array<T,N> result;
  for (size_t i = 0; i < N; ++i) {
    result[i] = b[i] == 0 ? 0 : a[i] / b[i];
  }
  return result;
}

template <typename T, size_t N>
std::array<T,N> multiplyArray(const std::array<T,N> &a, double scalar) {
  std::array<T,N> result;
  for (size_t i = 0; i < N; ++i) {
    result[i] = a[i] * scalar;
  }
  return result;
}

template <typename T, size_t N>
std::array<T,N> multiplyArrays(const std::array<T,N> &a, const std::array<T,N> &b) {
  std::array<T,N> result;
  for (size_t i = 0; i < N; ++i) {
    result[i] = a[i] * b[i];
  }
  return result;
}

template <typename T, size_t N>
std::array<T,N> absArray(const std::array<T,N> &a) {
  std::array<T,N> result;
  for (size_t i = 0; i < N; ++i) {
    result[i] = std::abs(a[i]);
  }
  return result;
}

template <typename T, size_t N>
std::array<T,N> subtractArrays(const std::array<T,N> &a, const std::array<T,N> &b) {
  std::array<T,N> result;
  for (size_t i = 0; i < N; ++i) {
    result[i] = a[i] - b[i];
  }
  return result;
}

template <typename T, size_t N>
bool allLessThan(const std::array<T,N> &a, const std::array<T,N> &b) {
  for (size_t i = 0; i < N; ++i) {
    if (a[i] >= b[i]) {
      return false;
    }
  }
  return true;
}


// Get Standard Deviation and Mean from Vectors
template <typename T, size_t N> 
std::array<T,N> getMean(
  const std::vector<std::array<T,N>> &vectors
) {
  size_t num_samples = vectors.size();
  size_t num_features = vectors[0].size();
  // Init in 0
  std::array<T,N> mean = std::array<T,N>();
  // Calculate mean
  for (size_t i = 0; i < num_samples; ++i) {
    mean = sumArrays(mean, vectors[i]);
  }
  mean = divideArray(mean, num_samples);
  return mean;
}

template <typename T, size_t N>
std::array<T,N> getStdDev(
  const std::vector<std::array<T,N>> &vectors,
  const std::array<T,N> &mean
) {
  size_t num_samples = vectors.size();
  size_t num_features = vectors[0].size();
  // Init in 0
  std::array<T,N> stdDev = std::array<T,N>();
  // Calculate stdDev
  for (size_t i = 0; i < num_samples; ++i) {
    stdDev = sumArrays(stdDev, multiplyArrays(vectors[i], vectors[i]));
  }
  for (size_t j = 0; j < num_features; ++j) {
    stdDev[j] = std::sqrt(stdDev[j] / num_samples - mean[j] * mean[j]);
  }
  return stdDev;
}

struct SampleData {
  // === SCREEN POSITION ===
  Point2f pFilm;
  // === COLORS ===
  Spectrum L;
  // === FEATURES ===
  // First intersection
  Normal3f n0; // Normal
  Point3f p0;  // World-space position
              // Texture?
  // Second intersection
  Normal3f n1; // Normal
  Point3f p1;  // World-space position
  // === RANDOM PARAMETERS ==
  Point2f pLens;
  // === OTHERS? ===
  Float rayWeight;
  // CONSTRUCTOR
  SampleData(
    const Point2f &pFilm,
    const Point2f &pLens,
    const Spectrum &L,
    Float rayWeight
  ):
    pFilm(pFilm),
    pLens(pLens),
    n0(Normal3f(0, 0, 0)),
    p0(Point3f(0, 0, 0)),
    n1(Normal3f(0, 0, 0)),
    p1(Point3f(0, 0, 0)),
    rayWeight(rayWeight),
    L(L)
    {};

  //  == TRANSFORM TO VECTORS ==
  SampleF toFeatureVector() const {
    return {
      n0.x, n0.y, n0.z,
      p0.x, p0.y, p0.z,
      n1.x, n1.y, n1.z,
      p1.x, p1.y, p1.z
    };
  }
  std::array<double, 3> toN0() const {
    return {n0.x, n0.y, n0.z};
  }
  std::array<double, 3> toN1() const {
    return {n1.x, n1.y, n1.z};
  }
  std::array<double, 3> toP0() const {
    return {p0.x, p0.y, p0.z};
  }
  std::array<double, 3> toP1() const {
    return {p1.x, p1.y, p1.z};
  }
  SampleP toPositionVector() const {
    return {pFilm.x, pFilm.y};
  }
  SampleR toRandomVector() const {
    return {pLens.x, pLens.y};
  }
};

// Declare SampleDataSet as a vector of SampleData
typedef std::vector<SampleData> SampleDataSet;
typedef std::vector<SampleDataSet> SampleDataSetVector;
typedef std::vector<std::vector<SampleDataSet>> SampleDataSetMatrix;

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
    void getStatsPerPixel(
      const SampleDataSetMatrix &samples,
      SampleFMatrix &meanMatrix,
      SampleFMatrix &stdDevMatrix
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
