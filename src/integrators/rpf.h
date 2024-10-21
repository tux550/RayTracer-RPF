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

  // + operator
  BasicRGB operator+(const BasicRGB &other) const {
    return BasicRGB(r + other.r, g + other.g, b + other.b);
  }
  // / operator for double
  BasicRGB operator/(double scalar) const {
    return BasicRGB(r / scalar, g / scalar, b / scalar);
  }
  // / operator for BasicRGB
  BasicRGB operator/(const BasicRGB &other) const {
    double nr = other.r == 0 ? 0 : r / other.r;
    double ng = other.g == 0 ? 0 : g / other.g;
    double nb = other.b == 0 ? 0 : b / other.b;
    return BasicRGB(nr, ng, nb);
  }
  // * operator for double
  BasicRGB operator*(double scalar) const {
    return BasicRGB(r * scalar, g * scalar, b * scalar);
  }
  // * operator for BasicRGB
  BasicRGB operator*(const BasicRGB &other) const {
    return BasicRGB(r * other.r, g * other.g, b * other.b);
  }

};

std::vector<std::vector<BasicRGB>> createBasicRGBMatrix(
  size_t nRows,
  size_t nCols
);


typedef std::vector<double> InfoVec;
typedef std::vector<std::vector<InfoVec>> InfoVecMatrix;

struct SampleFeatures {
  // FIRST INTERSECTION
  Normal3f n0; // Normal
  Point3f p0;  // World-space position
              // Texture?
  // SECOND INTERSECTION
  Normal3f n1; // Normal
  Point3f p1;  // World-space position
  // POSITION
  Point2f pFilm;
  Point2f pLens;
  // Others
  Float rayWeight;
  Spectrum L;

  // CONSTRUCTOR
  SampleFeatures(const Point2f &pFilm, const Point2f &pLens, const Spectrum &L, Float rayWeight):
    pFilm(pFilm),
    pLens(pLens),
    n0(Normal3f(0, 0, 0)),
    p0(Point3f(0, 0, 0)),
    n1(Normal3f(0, 0, 0)),
    p1(Point3f(0, 0, 0)),
    rayWeight(rayWeight),
    L(L)
    {};
  
  // To vector
  InfoVec toInfoVec() const {
    return {
      n0.x, n0.y, n0.z,
      p0.x, p0.y, p0.z,
      n1.x, n1.y, n1.z,
      p1.x, p1.y, p1.z,
      pFilm.x, pFilm.y,
      pLens.x, pLens.y,
      rayWeight
      //L.c[0], L.c[1], L.c[2]
    };
  }
};

// Get Standard Deviation and Mean from Vectors
InfoVec getMean(
  const std::vector<InfoVec> &vectors
);

InfoVec getStdDev(
  const std::vector<InfoVec> &vectors,
  const InfoVec &mean
);

void writeSFMat(const std::vector<std::vector<SampleFeatures>>  &sfMat, const std::string &filename);

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
      SampleFeatures &sf,
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
      const std::vector<std::vector<
        std::vector<SampleFeatures>
      >> &samples,
      InfoVecMatrix &meanMatrix,
      InfoVecMatrix &stdDevMatrix
    );
    std::vector<std::vector<
      std::vector<SampleFeatures>
    >> getNeighborhoodSamples(
      const std::vector<std::vector<
        std::vector<SampleFeatures>
      >> &samples,
      size_t box_size
    );
};

RPFIntegrator *CreateRPFIntegrator(
  const ParamSet &params,
  std::shared_ptr<Sampler> sampler,
  std::shared_ptr<const Camera> camera
); 

}



#endif  // PBRT_INTEGRATORS_RPF_PATH_H
