#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_SAMPLEDATA_H
#define PBRT_SAMPLEDATA_H

#include <array>
#include <vector>
#include "pbrt.h"
#include "geometry.h"
#include "spectrum.h"



namespace pbrt {

// SampleFeatures: n0, p0, n1, p1
typedef std::array<double, 12> SampleF;
typedef std::vector<SampleF> SampleFVector;
typedef std::vector<SampleFVector> SampleFMatrix;
// SamplePosition: pFilm
typedef std::array<double, 2> SampleP;
// SampleRandom: pLens
typedef std::array<double, 2> SampleR;

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

}  // namespace pbrt

#endif // PBRT_SAMPLEDATA_H