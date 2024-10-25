#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CUSTOM_SAMPLE_FILM_H
#define PBRT_CUSTOM_SAMPLE_FILM_H

#include <mutex>
#include <memory>
#include "pbrt.h"
#include "custom/sd.h"


namespace pbrt {

struct SamplingTile {
  Bounds2i pixelBounds;
  std::vector<std::vector<SampleDataSet>> samples;

  SamplingTile(const Bounds2i &pixelBounds);

  void addSample(const Point2i &pixel, const SampleData &sample);

  SampleDataSet getPixelSamples(const Point2i &pixel);
};

struct SamplingFilm {
  std::mutex mutex;
  std::vector<std::vector<SampleDataSet>> samples;
  Bounds2i pixelBounds;

  SamplingFilm(const Bounds2i &pixelBounds);

  void AddSample(const Point2i &pixel, const SampleData &sample);

  std::unique_ptr<SamplingTile> GetSamplingTile(const Bounds2i &sampleBounds);

  void MergeSamplingTile(std::unique_ptr<SamplingTile> tile);

};




} // namespace pbrt

#endif // PBRT_CUSTOM_SAMPLE_FILM_H