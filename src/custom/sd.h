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
// SampleColor: L
typedef std::array<double, 3> SampleC;

// Types
typedef std::array<double, 19> SampleFullArray;
typedef SampleFullArray SampleA;
typedef std::vector<SampleFullArray> SampleAVector;
typedef std::vector<SampleAVector> SampleAMatrix;

struct SampleData {

  // Data regions: 
  // SCREEN POSITION: pFilm
  // COLORS: L
  // RANDOM PARAMETERS: pLens
  // FEATURES: n0, p0, n1, p1 (+Texture?)
  // Other? rayWeight 
  SampleFullArray data;
  Float rayWeight;

  void setPFilm(const Point2f &pFilm) {
    data[0] = pFilm.x;
    data[1] = pFilm.y;
  }
  void setL(const Spectrum &L) {
    data[2] = L[0];
    data[3] = L[1];
    data[4] = L[2];
  }
  void setPLens(const Point2f &pLens) {
    data[5] = pLens.x;
    data[6] = pLens.y;
  }
  void setN0(const Normal3f &n0) {
    data[7] = n0.x;
    data[8] = n0.y;
    data[9] = n0.z;
  }
  void setP0(const Point3f &p0) {
    data[10] = p0.x;
    data[11] = p0.y;
    data[12] = p0.z;
  }
  void setN1(const Normal3f &n1) {
    data[13] = n1.x;
    data[14] = n1.y;
    data[15] = n1.z;
  }
  void setP1(const Point3f &p1) {
    data[16] = p1.x;
    data[17] = p1.y;
    data[18] = p1.z;
  }
  // === GETTERS ===
  Point2f getPFilm() const {
    return Point2f(data[0], data[1]);
  }
  Point2f getPLens() const {
    return Point2f(data[5], data[6]);
  }
  Spectrum getL() const {
    Float rgb[3] = {data[2], data[3], data[4]};
    return Spectrum::FromRGB(rgb);
  }
  Normal3f getN0() const {
    return Normal3f(data[7], data[8], data[9]);
  }
  Point3f getP0() const {
    return Point3f(data[10], data[11], data[12]);
  }
  Normal3f getN1() const {
    return Normal3f(data[13], data[14], data[15]);
  }
  Point3f getP1() const {
    return Point3f(data[16], data[17], data[18]);
  }

  // CONSTRUCTORS
  SampleData():
    data(),
    rayWeight(0)
    {
      for (int i = 0; i < 19; i++) {
        data[i] = 0;
      }
    };
  
  SampleData(
    const Point2f &pFilm,
    const Point2f &pLens,
    const Spectrum &L,
    Float rayWeight
  ):
    data(),
    rayWeight(rayWeight)
    {
      setPFilm(pFilm);
      setL(L);
      setPLens(pLens);
    };

  //  == EXTRACT SECTIONS ==
  SampleF getFeatures() const {
    return {
      data[7], data[8], data[9], // n0
      data[10], data[11], data[12], // p0
      data[13], data[14], data[15], // n1
      data[16], data[17], data[18] // p1
    };
  }

  SampleP getPosition() const {
    return {data[0], data[1]};
  }

  SampleR getRandom() const {
    return {data[5], data[6]};
  }

  SampleC getColor() const {
    return {data[2], data[3], data[4]};
  }

  SampleFullArray getFullArray() const {
    return data;
  }
  // Set from full array
  void setFullArray(const SampleFullArray &fullArray) {
    data = fullArray;
  }

  // Normalize
  SampleData normalized(const SampleFullArray &mean, const SampleFullArray &stdDev) const {
    SampleData normalizedData;
    for (int i = 0; i < 19; i++) {
      normalizedData.data[i] = (data[i] - mean[i]) / stdDev[i];
    }
    normalizedData.rayWeight = rayWeight;
    return normalizedData;
  }
};

// Declare SampleDataSet as a vector of SampleData
typedef std::vector<SampleData> SampleDataSet;
typedef std::vector<SampleDataSet> SampleDataSetVector;
typedef std::vector<std::vector<SampleDataSet>> SampleDataSetMatrix;

}  // namespace pbrt

#endif // PBRT_SAMPLEDATA_H