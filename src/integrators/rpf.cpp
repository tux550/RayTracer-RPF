
// rpf.cpp*
#include "integrators/rpf.h"
#include "bssrdf.h"
#include "camera.h"
#include "film.h"
#include "interaction.h"
#include "paramset.h"
#include "scene.h"
#include "stats.h"

#include "integrator.h"
#include "sampler.h"
#include "progressreporter.h"
#include "lightdistrib.h"

namespace pbrt {
  STAT_COUNTER("Integrator/Camera rays traced", nCameraRays);
  STAT_COUNTER("Integrator/MaxNeighborhoodSize", maxNeighborhoodSize);
  STAT_COUNTER("Integrator/MeanNeighborhoodSize", meanNeighborhoodSize);
  STAT_COUNTER("Integrator/MinNeighborhoodSize", minNeighborhoodSize);
  STAT_COUNTER("Integrator/EmptyNeighborhoods", emptyNeighborhoods);

  STAT_PERCENT("Integrator/Zero-radiance paths", zeroRadiancePaths, totalPaths);
  STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);

  std::vector<std::vector<BasicRGB>> createBasicRGBMatrix(
    size_t nRows,
    size_t nCols
  ) {
    return std::vector<std::vector<BasicRGB>>(
      nRows,
      std::vector<BasicRGB>(nCols, {0, 0, 0})
    );
  }

  InfoVec getMean(
    const std::vector<InfoVec> &vectors
  ) {
    size_t num_samples = vectors.size();
    size_t num_features = vectors[0].size();
    // Init in 0
    auto mean = InfoVec(num_features, 0);
    // Calculate mean
    for (size_t i = 0; i < num_samples; ++i) {
      for (size_t j = 0; j < num_features; ++j) {
        mean[j] += vectors[i][j];
      }
    }
    for (size_t j = 0; j < num_features; ++j) {
      mean[j] /= num_samples;
    }
    return mean;
  }

  InfoVec getStdDev(
    const std::vector<InfoVec> &vectors,
    const InfoVec &mean
  ) {
    size_t num_samples = vectors.size();
    size_t num_features = vectors[0].size();
    // Init in 0
    InfoVec stdDev = InfoVec(num_features, 0);
    // Calculate stdDev
    for (size_t i = 0; i < num_samples; ++i) {
      for (size_t j = 0; j < num_features; ++j) {
        stdDev[j] += (vectors[i][j] - mean[j]) * (vectors[i][j] - mean[j]);
      }
    }
    for (size_t j = 0; j < num_features; ++j) {
      stdDev[j] = std::sqrt(stdDev[j] / num_samples);
    }
    return stdDev;
  }
  
  void writeSFMat(const std::vector<std::vector<std::vector<SampleFeatures>>>  &sfMat, const std::string &filename) {
    // Remove any extension from filename
    std::string base_filename = filename;
    std::string::size_type idx = base_filename.rfind('.');
    if (idx != std::string::npos) {
      base_filename = base_filename.substr(0, idx);
    }

    std::string filmPosFilename = base_filename + "_Film_Position.exr";
    std::string lensPosFilename = base_filename + "_Lens_Position.exr";
    std::string n0filename = base_filename + "_I0_Normal.exr";
    std::string n1filename = base_filename + "_I1_Normal.exr";
    std::string p0filename = base_filename + "_I0_Position.exr";
    std::string p1filename = base_filename + "_I1_Position.exr";

    size_t nRows = sfMat.size();
    size_t nCols = sfMat[0].size();
    // Calculate magnitude of normal vectors and position vectors
    auto n0Mat = createBasicRGBMatrix(nRows, nCols);
    auto n1Mat = createBasicRGBMatrix(nRows, nCols);
    auto p0Mat = createBasicRGBMatrix(nRows, nCols);
    auto p1Mat = createBasicRGBMatrix(nRows, nCols);
    auto filmPosMat = createBasicRGBMatrix(nRows, nCols);
    auto lensPosMat = createBasicRGBMatrix(nRows, nCols);
    
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        for (const SampleFeatures &sf : sfMat[i][j]) {
          n0Mat[i][j] = n0Mat[i][j] + BasicRGB(sf.n0.x, sf.n0.y, sf.n0.z);
          n1Mat[i][j] = n1Mat[i][j] + BasicRGB(sf.n1.x, sf.n1.y, sf.n1.z);
          p0Mat[i][j] = p0Mat[i][j] + BasicRGB(sf.p0.x, sf.p0.y, sf.p0.z);
          p1Mat[i][j] = p1Mat[i][j] + BasicRGB(sf.p1.x, sf.p1.y, sf.p1.z);
          filmPosMat[i][j] = filmPosMat[i][j] + BasicRGB(sf.pFilm.x, sf.pFilm.y, 0);
          lensPosMat[i][j] = lensPosMat[i][j] + BasicRGB(sf.pLens.x, sf.pLens.y, 0);
        }
        n0Mat[i][j] = n0Mat[i][j] / sfMat[i][j].size();
        n1Mat[i][j] = n1Mat[i][j] / sfMat[i][j].size();
        p0Mat[i][j] = p0Mat[i][j] / sfMat[i][j].size();
        p1Mat[i][j] = p1Mat[i][j] / sfMat[i][j].size();
        filmPosMat[i][j] = filmPosMat[i][j] / sfMat[i][j].size();
        lensPosMat[i][j] = lensPosMat[i][j] / sfMat[i][j].size();
      }
    }
    // Normalize
    BasicRGB maxN0 = {0, 0, 0};
    BasicRGB maxN1 = {0, 0, 0};
    BasicRGB maxP0 = {0, 0, 0};
    BasicRGB maxP1 = {0, 0, 0};
    BasicRGB maxFilmPos = {0, 0, 0};
    BasicRGB maxLensPos = {0, 0, 0};

    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        maxN0.r = std::max(maxN0.r, n0Mat[i][j].r);
        maxN0.g = std::max(maxN0.g, n0Mat[i][j].g);
        maxN0.b = std::max(maxN0.b, n0Mat[i][j].b);
        maxN1.r = std::max(maxN1.r, n1Mat[i][j].r);
        maxN1.g = std::max(maxN1.g, n1Mat[i][j].g);
        maxN1.b = std::max(maxN1.b, n1Mat[i][j].b);
        maxP0.r = std::max(maxP0.r, p0Mat[i][j].r);
        maxP0.g = std::max(maxP0.g, p0Mat[i][j].g);
        maxP0.b = std::max(maxP0.b, p0Mat[i][j].b);
        maxP1.r = std::max(maxP1.r, p1Mat[i][j].r);
        maxP1.g = std::max(maxP1.g, p1Mat[i][j].g);
        maxP1.b = std::max(maxP1.b, p1Mat[i][j].b);
        maxFilmPos.r = std::max(maxFilmPos.r, filmPosMat[i][j].r);
        maxFilmPos.g = std::max(maxFilmPos.g, filmPosMat[i][j].g);
        maxLensPos.r = std::max(maxLensPos.r, lensPosMat[i][j].r);
        maxLensPos.g = std::max(maxLensPos.g, lensPosMat[i][j].g);
      }
    }
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        n0Mat[i][j] = (n0Mat[i][j] / maxN0);
        n1Mat[i][j] = (n1Mat[i][j] / maxN1);
        p0Mat[i][j] = (p0Mat[i][j] / maxP0);
        p1Mat[i][j] = (p1Mat[i][j] / maxP1);
        filmPosMat[i][j] = (filmPosMat[i][j] / maxFilmPos);
        lensPosMat[i][j] = (lensPosMat[i][j] / maxLensPos);
      }
    }

    // Write to file
    Imf::Rgba* filmPosPixels = new Imf::Rgba[nCols * nRows];
    Imf::Rgba* lensPosPixels = new Imf::Rgba[nCols * nRows];
    Imf::Rgba* n0pixels = new Imf::Rgba[nCols * nRows];
    Imf::Rgba* n1pixels = new Imf::Rgba[nCols * nRows];
    Imf::Rgba* p0pixels = new Imf::Rgba[nCols * nRows];
    Imf::Rgba* p1pixels = new Imf::Rgba[nCols * nRows];
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        n0pixels[j * nRows + i] = Imf::Rgba(n0Mat[i][j].r, n0Mat[i][j].g, n0Mat[i][j].b, 1);
        n1pixels[j * nRows + i] = Imf::Rgba(n1Mat[i][j].r, n1Mat[i][j].g, n1Mat[i][j].b, 1);
        p0pixels[j * nRows + i] = Imf::Rgba(p0Mat[i][j].r, p0Mat[i][j].g, p0Mat[i][j].b, 1);
        p1pixels[j * nRows + i] = Imf::Rgba(p1Mat[i][j].r, p1Mat[i][j].g, p1Mat[i][j].b, 1);
        // Film and Lens position
        filmPosPixels[j * nRows + i] = Imf::Rgba(filmPosMat[i][j].r, filmPosMat[i][j].g, 0, 1);
        lensPosPixels[j * nRows + i] = Imf::Rgba(lensPosMat[i][j].r, lensPosMat[i][j].g, 0, 1);

      }
    }
    Imf::RgbaOutputFile n0file(n0filename.c_str(), nCols, nRows, Imf::WRITE_RGBA);
    n0file.setFrameBuffer(n0pixels, 1, nCols);
    n0file.writePixels(nRows);
    Imf::RgbaOutputFile n1file(n1filename.c_str(), nCols, nRows, Imf::WRITE_RGBA);
    n1file.setFrameBuffer(n1pixels, 1, nCols);
    n1file.writePixels(nRows);
    Imf::RgbaOutputFile p0file(p0filename.c_str(), nCols, nRows, Imf::WRITE_RGBA);
    p0file.setFrameBuffer(p0pixels, 1, nCols);
    p0file.writePixels(nRows);
    Imf::RgbaOutputFile p1file(p1filename.c_str(), nCols, nRows, Imf::WRITE_RGBA);
    p1file.setFrameBuffer(p1pixels, 1, nCols);
    p1file.writePixels(nRows);
    // Film and Lens position
    Imf::RgbaOutputFile filmPosFile(filmPosFilename.c_str(), nCols, nRows, Imf::WRITE_RGBA);
    filmPosFile.setFrameBuffer(filmPosPixels, 1, nCols);
    filmPosFile.writePixels(nRows);
    Imf::RgbaOutputFile lensPosFile(lensPosFilename.c_str(), nCols, nRows, Imf::WRITE_RGBA);
    lensPosFile.setFrameBuffer(lensPosPixels, 1, nCols);
    lensPosFile.writePixels(nRows);

    delete[] filmPosPixels;
    delete[] lensPosPixels;
    delete[] p0pixels;
    delete[] p1pixels;
    delete[] n0pixels;
    delete[] n1pixels;
  }

  RPFIntegrator::RPFIntegrator(
    int maxDepth,
    std::shared_ptr<const Camera> camera,
    std::shared_ptr<Sampler> sampler,
    const Bounds2i &pixelBounds,
    Float rrThreshold,
    const std::string &lightSampleStrategy
  ) : 
    camera(camera),
    sampler(sampler),
    pixelBounds(pixelBounds),
    maxDepth(maxDepth),
    rrThreshold(rrThreshold),
    lightSampleStrategy(lightSampleStrategy) {}



  void RPFIntegrator::Preprocess(const Scene &scene, Sampler &sampler) {
    lightDistribution = CreateLightSampleDistribution(lightSampleStrategy, scene);
  }

  // Preprocess samples
  void RPFIntegrator::getStatsPerPixel(
    const std::vector<std::vector<
      std::vector<SampleFeatures>
    >> &samples,
    InfoVecMatrix &meanMatrix,
    InfoVecMatrix &stdDevMatrix
  ) {
    size_t nRows = samples.size();
    size_t nCols = samples[0].size();
    size_t nSamples = samples[0][0].size();
    // Init matrices (nRows x nCols)
    meanMatrix = InfoVecMatrix(nRows, std::vector<InfoVec>(nCols, InfoVec()));
    stdDevMatrix = InfoVecMatrix(nRows, std::vector<InfoVec>(nCols, InfoVec()));
    // Calculate mean and stdDev for each pixel
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        // Get mean and stdDev for each feature
        std::vector<InfoVec> vectors;
        for (const SampleFeatures &sf : samples[i][j]) {
          vectors.push_back(sf.toInfoVec());
        }
        meanMatrix[i][j] = getMean(vectors);
        stdDevMatrix[i][j] = getStdDev(vectors, meanMatrix[i][j]);
      }
    }
  }

  std::vector<std::vector<
    std::vector<SampleFeatures>
  >> RPFIntegrator::getNeighborhoodSamples(
    const std::vector<std::vector<
      std::vector<SampleFeatures>
    >> &samples,
    size_t box_size
  ) {
    // GET STATS FOR EACH PIXEL
    // Init matrices
    auto meanMatrix = InfoVecMatrix();
    auto stdDevMatrix = InfoVecMatrix();
    // Get mean and stdDev for each pixel
    getStatsPerPixel(samples, meanMatrix, stdDevMatrix);
    // COLLECT NEIGHBORHOOD SAMPLES THAT ARE WITHIN 3 STD DEVIATIONS
    size_t nRows = samples.size();
    size_t nCols = samples[0].size();
    auto neighborhoodSamples = std::vector<std::vector<std::vector<SampleFeatures>>>();
    for (size_t i = 0; i < nRows; ++i) {
      std::vector<std::vector<SampleFeatures>> row;
      for (size_t j = 0; j < nCols; ++j) {
        // Current pixel (i, j). 
        // Check all pixels within box_size (not including current pixel)
        // Assumes box_size is odd
        auto b_delta = (box_size-1) / 2;
        auto neighborhood = std::vector<SampleFeatures>();
        for (size_t x = i - b_delta; x <= i + b_delta; ++x) {
          for (size_t y = j - b_delta; y <= j + b_delta; ++y) {
            // Skip if current pixel
            if (x == i && y == j) {
              continue;
            }
            // If pixel is outside bounds, skip
            if (x < 0 || x >= nRows || y < 0 || y >= nCols) {
              continue;
            }
            // Check each sample in pixel
            for (const SampleFeatures &sf : samples[x][y]) {
              // Check if all features are within 3 std deviations
              bool within3StdDevs = true;
              for (size_t k = 0; k < sf.toInfoVec().size(); ++k) {
                if (std::abs(sf.toInfoVec()[k] - meanMatrix[i][j][k]) > 3 * stdDevMatrix[i][j][k]) {
                  within3StdDevs = false;
                  break;
                }
              }
              if (within3StdDevs) {
                neighborhood.push_back(sf);
              }
            }
          }
        }
        // Add neighborhood to row
        row.push_back(neighborhood);
      }
      // Add row to neighborhoodSamples
      neighborhoodSamples.push_back(row);
    }
    // Create a new matrix, containing a set of neighbour samples for each pixel  
    return neighborhoodSamples;
  }

  // Render
  void RPFIntegrator::Render(const Scene &scene) {  
    Bounds2i sampleBounds = camera->film->GetSampleBounds();
    Vector2i sampleExtent = sampleBounds.Diagonal();
    ProgressReporter reporter(sampleExtent.x*sampleExtent.y, "Rendering");
    // Allocate 3D matrix of samples
    std::vector<std::vector<std::vector<SampleFeatures>>> samples(
      sampleExtent.x,
      std::vector<std::vector<SampleFeatures>>(
        sampleExtent.y,
        std::vector<SampleFeatures>()
      )
    );
    { 
      // Allocate _MemoryArena_
      MemoryArena arena;
      // Compute sample bounds for tile
      int x0 = sampleBounds.pMin.x;
      int x1 = sampleBounds.pMax.x;
      int y0 = sampleBounds.pMin.y;
      int y1 = sampleBounds.pMax.y;
      // Loop over pixels to render them
      for (Point2i pixel : sampleBounds) {
        // Init pixel
        {
          ProfilePhase pp(Prof::StartPixel);
          sampler->StartPixel(pixel);
        }
        // For RNG reproducibility
        if (!InsideExclusive(pixel, pixelBounds))
          continue;

        do {
          // Initialize _CameraSample_ for current sample
          CameraSample cameraSample = sampler->GetCameraSample(pixel);
          // Generate camera ray for current sample
          RayDifferential ray;
          Float rayWeight = camera->GenerateRayDifferential(cameraSample, &ray);
          ray.ScaleDifferentials(1 / std::sqrt((Float)sampler->samplesPerPixel));
          ++nCameraRays;
          // Evaluate radiance along camera ray and capture Features
          SampleFeatures sf(
            cameraSample.pFilm, // Film position
            cameraSample.pLens, // Lens position
            0.f ,                // L (placeholder)
            rayWeight           // Ray weight
          );
          if (rayWeight > 0) {
            // Evaluate radiance along camera ray
            Li(ray, scene, *sampler, arena, sf);
          }
          samples[pixel.x][pixel.y].push_back(sf);
          // Free _MemoryArena_ memory from computing image sample value
          arena.Reset();
        } while (sampler->StartNextSample());        
      }
    }
    LOG(INFO) << "Finished sampling pixels" << sampleBounds;
    // Write FeatureVector data to file
    writeSFMat(samples, camera->film->filename);


    // PREPROCESSING THE SAMPLES
    // To do this, we compute the average feature vector m{f,p} and the
    // standard deviation vector σf P for each component of the feature
    // vector for the set of samples P at the current pixel. We then cre-
    // ate our neighborhood N using only samples whose features are all
    // within 3 standard deviations of the mean for the pixel
    // Get Neighborhood samples
    auto neighborhoodSamples = getNeighborhoodSamples(samples, 3);
    // COUT STATS: min, max and average number of samples in the neighborhood
    size_t minSamples = std::numeric_limits<size_t>::max();
    size_t maxSamples = 0;
    size_t totalSamples = 0;
    for (size_t i = 0; i < sampleExtent.x; ++i) {
      for (size_t j = 0; j < sampleExtent.y; ++j) {
        size_t nSamples = neighborhoodSamples[i][j].size();
        minSamples = std::min(minSamples, nSamples);
        maxSamples = std::max(maxSamples, nSamples);
        totalSamples += nSamples;
      }
    };

    minNeighborhoodSize = minSamples;
    maxNeighborhoodSize = maxSamples;
    meanNeighborhoodSize = totalSamples / (sampleExtent.x * sampleExtent.y);
    emptyNeighborhoods = minSamples == 0 ? 1 : 0;
    


    // 1. Clustering
    // 2. Normalization

    // STATISTICAL DEPENDENCY ESTIMATION
    // 1. joint mutual information for each feature and random parameter
    // 2. calculate filter weights alpha and beta for each feature
    // 3. calculate mutual information

    // FILTERING THE SAMPLES














    // Get filmTile
    std::unique_ptr<FilmTile> filmTile = camera->film->GetFilmTile(sampleBounds);
    // Add camera ray's contribution to image
    double numSamples = 0;
    for (int x = 0; x < sampleExtent.x; ++x) {
      for (int y = 0; y < sampleExtent.y; ++y) {
        for (const SampleFeatures &sf : samples[x][y]) {
          filmTile->AddSample(sf.pFilm, sf.L, sf.rayWeight);
        }
        numSamples += samples[x][y].size();
      }
    }
    // Sample index
    size_t sx = samples.size() / 2;
    size_t sy = samples[0].size() / 2;

    // Merge image tile into _Film_
    camera->film->MergeFilmTile(std::move(filmTile));
    LOG(INFO) << "Rendering finished";

    // Save final image after rendering
    camera->film->WriteImage();
  }

  // Luminance samplings
  void RPFIntegrator::Li(
    const RayDifferential &r, const Scene &scene,
    Sampler &sampler, MemoryArena &arena,
    SampleFeatures &sf,
    int depth
  ) const {

    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L(0.f), beta(1.f);
 
    RayDifferential ray(r);
    bool specularBounce = false;
    int bounces;
    Float etaScale = 1;

    for (bounces = 0;; ++bounces) {
      // Find next path vertex and accumulate contribution
      VLOG(2) << "Path tracer bounce " << bounces << ", current L = " << L
              << ", beta = " << beta;

      // Intersect _ray_ with scene and store intersection in _isect_
      SurfaceInteraction isect;
      bool foundIntersection = scene.Intersect(ray, &isect);

      // Possibly add emitted light at intersection
      if (bounces == 0 || specularBounce) {
        // Add emitted light at path vertex or from the environment
        if (foundIntersection) {
          L += beta * isect.Le(-ray.d);
          VLOG(2) << "Added Le -> L = " << L;
        } else {
          for (const auto &light : scene.infiniteLights) {
            L += beta * light->Le(ray);
          }
        }
      }

      // Terminate path if ray escaped or _maxDepth_ was reached
      if (!foundIntersection || bounces >= maxDepth) {
        break;
      }
      
      // EDIT: Save FeatureVector data
      if (bounces == 0) {
        sf.n0 = isect.n;
        sf.p0 = isect.p;
      } else if (bounces == 1) {
        sf.n1 = isect.n;
        sf.p1 = isect.p;
      }

      // Compute scattering functions and skip over medium boundaries
      isect.ComputeScatteringFunctions(ray, arena, true);
      if (!isect.bsdf) {
        ray = isect.SpawnRay(ray.d);
        --bounces;
        continue;
      }

      // Sample illumination from lights to find path contribution
      // (But skip this for perfectly specular BSDFs.)
      if (isect.bsdf->NumComponents(BxDFType(BSDF_ALL & ~BSDF_SPECULAR)) > 0) {
        ++totalPaths;
        Spectrum Ld = beta * UniformSampleOneLight(isect, scene, arena, sampler, false, nullptr);
        VLOG(2) << "Sampled direct lighting Ld = " << Ld;
        if (Ld.IsBlack()) {
          ++zeroRadiancePaths;
        }
        CHECK_GE(Ld.y(), 0.f);
        L += Ld;
      }

      // Sample BSDF to get new path direction
      Vector3f wo = -ray.d, wi;
      Float pdf;
      BxDFType flags;
      Spectrum f = isect.bsdf->Sample_f(wo, &wi, sampler.Get2D(), &pdf, BSDF_ALL, &flags);
      VLOG(2) << "Sampled BSDF, f = " << f << ", pdf = " << pdf;
      if (f.IsBlack() || pdf == 0.f) break;
      beta *= f * AbsDot(wi, isect.shading.n) / pdf;
      VLOG(2) << "Updated beta = " << beta;
      CHECK_GE(beta.y(), 0.f);
      DCHECK(!std::isinf(beta.y()));
      specularBounce = (flags & BSDF_SPECULAR) != 0;
      if ((flags & BSDF_SPECULAR) && (flags & BSDF_TRANSMISSION)) {
          Float eta = isect.bsdf->eta;
          // Update the term that tracks radiance scaling for refraction
          // depending on whether the ray is entering or leaving the
          // medium.
          etaScale *= (Dot(wo, isect.n) > 0) ? (eta * eta) : 1 / (eta * eta);
      }
      ray = isect.SpawnRay(wi);

      // Account for subsurface scattering, if applicable
      if (isect.bssrdf && (flags & BSDF_TRANSMISSION)) {
          // Importance sample the BSSRDF
          SurfaceInteraction pi;
          Spectrum S = isect.bssrdf->Sample_S(
              scene, sampler.Get1D(), sampler.Get2D(), arena, &pi, &pdf);
          DCHECK(!std::isinf(beta.y()));
          if (S.IsBlack() || pdf == 0) break;
          beta *= S / pdf;

          // Account for the direct subsurface scattering component
          L += beta * UniformSampleOneLight(pi, scene, arena, sampler, false,
                                            lightDistribution->Lookup(pi.p));

          // Account for the indirect subsurface scattering component
          Spectrum f = pi.bsdf->Sample_f(pi.wo, &wi, sampler.Get2D(), &pdf,
                                          BSDF_ALL, &flags);
          if (f.IsBlack() || pdf == 0) break;
          beta *= f * AbsDot(wi, pi.shading.n) / pdf;
          DCHECK(!std::isinf(beta.y()));
          specularBounce = (flags & BSDF_SPECULAR) != 0;
          ray = pi.SpawnRay(wi);
      }

      // Possibly terminate the path with Russian roulette.
      // Factor out radiance scaling due to refraction in rrBeta.
      Spectrum rrBeta = beta * etaScale;
      if (rrBeta.MaxComponentValue() < rrThreshold && bounces > 3) {
          Float q = std::max((Float).05, 1 - rrBeta.MaxComponentValue());
          if (sampler.Get1D() < q) break;
          beta /= 1 - q;
          DCHECK(!std::isinf(beta.y()));
      }
  }
  ReportValue(pathLength, bounces);
  // Set the Luminance value
  sf.L = L;
}

RPFIntegrator *CreateRPFIntegrator(
  const ParamSet &params,
  std::shared_ptr<Sampler> sampler,
  std::shared_ptr<const Camera> camera
) {
    int maxDepth = params.FindOneInt("maxdepth", 5);
    int np;
    const int *pb = params.FindInt("pixelbounds", &np);
    Bounds2i pixelBounds = camera->film->GetSampleBounds();
    if (pb) {
        if (np != 4)
            Error("Expected four values for \"pixelbounds\" parameter. Got %d.",
                  np);
        else {
            pixelBounds = Intersect(pixelBounds,
                                    Bounds2i{{pb[0], pb[2]}, {pb[1], pb[3]}});
            if (pixelBounds.Area() == 0)
                Error("Degenerate \"pixelbounds\" specified.");
        }
    }
    Float rrThreshold = params.FindOneFloat("rrthreshold", 1.);
    std::string lightStrategy =
        params.FindOneString("lightsamplestrategy", "spatial");
    return new RPFIntegrator(maxDepth, camera, sampler, pixelBounds,
                              rrThreshold, lightStrategy);
}

} // namespace pbrt