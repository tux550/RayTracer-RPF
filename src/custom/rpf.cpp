
// custom/rpf.cpp*
#include "custom/rpf.h"
#include "custom/sd.h"
#include "custom/mi.h"

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
  STAT_INT_DISTRIBUTION("Integrator/Samples Per Pixel", samplesPerPixel);
  STAT_INT_DISTRIBUTION("Integrator/Neighborhood Size", neighborhoodSize);

  STAT_PERCENT("Integrator/Zero-radiance paths", zeroRadiancePaths, totalPaths);
  STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);

  
  void visualizeSF(
    const SamplingFilm &sampling_film,
    const std::string &filename) {

    auto sdMat = sampling_film.samples;

    // Remove any extension from filename
    std::string base_filename = filename;
    std::string::size_type idx = base_filename.rfind('.');
    if (idx != std::string::npos) {
      base_filename = base_filename.substr(0, idx);
    }
    size_t nRows = sdMat.size();
    size_t nCols = sdMat[0].size();
    // Generate matrix
    auto n0Mat = createBasicRGBMatrix(nRows, nCols);
    auto n1Mat = createBasicRGBMatrix(nRows, nCols);
    auto p0Mat = createBasicRGBMatrix(nRows, nCols);
    auto p1Mat = createBasicRGBMatrix(nRows, nCols);
    auto filmPosMat = createBasicRGBMatrix(nRows, nCols);
    auto lensPosMat = createBasicRGBMatrix(nRows, nCols);
    // Fill matrices
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        for (const SampleData &sf : sdMat[i][j]) {
          auto n0 = sf.getN0();
          auto n1 = sf.getN1();
          auto p0 = sf.getP0();
          auto p1 = sf.getP1();
          auto pFilm = sf.getPFilm();
          auto pLens = sf.getPLens();

          n0Mat[i][j] = n0Mat[i][j] + BasicRGB(n0[0], n0[1], n0[2]);
          n1Mat[i][j] = n1Mat[i][j] + BasicRGB(n1[0], n1[1], n1[2]);
          p0Mat[i][j] = p0Mat[i][j] + BasicRGB(p0[0], p0[1], p0[2]);
          p1Mat[i][j] = p1Mat[i][j] + BasicRGB(p1[0], p1[1], p1[2]);
          filmPosMat[i][j] = filmPosMat[i][j] + BasicRGB(pFilm[0], pFilm[1], 0);
          lensPosMat[i][j] = lensPosMat[i][j] + BasicRGB(pLens[0], pLens[1], 0);
        }
        // Average
        if (sdMat[i][j].size() > 0) {
          n0Mat[i][j] = n0Mat[i][j] / sdMat[i][j].size();
          n1Mat[i][j] = n1Mat[i][j] / sdMat[i][j].size();
          p0Mat[i][j] = p0Mat[i][j] / sdMat[i][j].size();
          p1Mat[i][j] = p1Mat[i][j] / sdMat[i][j].size();
          filmPosMat[i][j] = filmPosMat[i][j] / sdMat[i][j].size();
          lensPosMat[i][j] = lensPosMat[i][j] / sdMat[i][j].size();
        }
      }
    }
    // Normalize
    normalizeRGBMatrix(n0Mat);
    normalizeRGBMatrix(n1Mat);
    normalizeRGBMatrix(p0Mat);
    normalizeRGBMatrix(p1Mat);
    normalizeRGBMatrix(filmPosMat);
    normalizeRGBMatrix(lensPosMat);
    // Write to file
    writeRGBMatrix(n0Mat, base_filename + "_I0_Normal.exr");
    writeRGBMatrix(n1Mat, base_filename + "_I1_Normal.exr");
    writeRGBMatrix(p0Mat, base_filename + "_I0_Position.exr");
    writeRGBMatrix(p1Mat, base_filename + "_I1_Position.exr");
    writeRGBMatrix(filmPosMat, base_filename + "_Film_Position.exr");
    writeRGBMatrix(lensPosMat, base_filename + "_Lens_Position.exr");
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
  void RPFIntegrator::getXStatsPerPixel(
    const SampleDataSetMatrix &samples,
    SampleXMatrix &meanMatrix,
    SampleXMatrix &stdDevMatrix
  ) {
    size_t nRows = samples.size();
    size_t nCols = samples[0].size();
    size_t nSamples = samples[0][0].size();
    // Init matrices (nRows x nCols)
    meanMatrix = SampleXMatrix(
      nRows,
      SampleXVector(nCols, SampleX())
    );
    stdDevMatrix = SampleXMatrix(
      nRows,
      SampleXVector(nCols, SampleX())
    );
    // Calculate mean and stdDev for each pixel
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        // Get mean and stdDev for each feature
        std::vector<SampleX> vectors;
        for (const SampleData &sd : samples[i][j]) {
          vectors.push_back(sd.getFullArray());
        }
        meanMatrix[i][j] = getMean(vectors);
        stdDevMatrix[i][j] = getStdDev(vectors, meanMatrix[i][j]);
      }
    }
  }

  SampleDataSetMatrix RPFIntegrator::getNeighborhoodSamples(
    const SampleDataSetMatrix &samples,
    const SampleFMatrix &meanMatrix,
    const SampleFMatrix &stdDevMatrix,
    size_t box_size // Always odd
  ) {
    // COLLECT NEIGHBORHOOD SAMPLES THAT ARE WITHIN 3 STD DEVIATIONS
    size_t nRows = samples.size();
    size_t nCols = samples[0].size();
    auto b_delta = (box_size-1) / 2; // Assumes box_size is odd
    auto neighborhoodSamples = SampleDataSetMatrix();
    for (size_t i = 0; i < nRows; ++i) {
      SampleDataSetVector row;
      for (size_t j = 0; j < nCols; ++j) {
        // Current pixel (i, j). 
        auto neighborhood = SampleDataSet();
        // Push all of current pixel's to neighborhood
        for (const SampleData &sf : samples[i][j]) {
          neighborhood.push_back(sf);
        }
        // Check all pixels within box_size (not including current pixel). If within 3 std devs, add to neighborhood
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
            for (const SampleData &sf : samples[x][y]) {
              // Check if all features are within 3 std deviations
              auto sfVec = sf.getFeatures();
              bool within3StdDevs = 
                allLessThan(
                  absArray(subtractArrays(sfVec, meanMatrix[i][j])),
                  multiplyArray(stdDevMatrix[i][j], 3) // 3 std devs
                );
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









void RPFIntegrator::FillSampleFilm(
  SamplingFilm &samplingFilm,
  const Scene &scene,
  const int tileSize) {
  // Get bounds
  Bounds2i sampleBounds = camera->film->GetSampleBounds();
  Vector2i sampleExtent = pixelBounds.Diagonal();
  // Divide into tiles
  Point2i nTiles(
    (sampleExtent.x + tileSize - 1) / tileSize,
    (sampleExtent.y + tileSize - 1) / tileSize
  );
  // Progress reporter
  ProgressReporter reporter_sampling(nTiles.x * nTiles.y, "Sampling");
  {
    ParallelFor2D([&](Point2i tile) {
      // Allocate
      MemoryArena arena;

      // Get sampler instance for tile
      int seed = tile.y * nTiles.x + tile.x;
      std::unique_ptr<Sampler> tileSampler = sampler->Clone(seed);

      // Compute sample bounds for tile
      int x0 = sampleBounds.pMin.x + tile.x * tileSize;
      int x1 = std::min(x0 + tileSize, sampleBounds.pMax.x);
      int y0 = sampleBounds.pMin.y + tile.y * tileSize;
      int y1 = std::min(y0 + tileSize, sampleBounds.pMax.y);
      Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));
      LOG(INFO) << "Starting image tile " << tileBounds;

      // Get SamplingTile for tile
      std::unique_ptr<SamplingTile> samplingTile = samplingFilm.GetSamplingTile(tileBounds);

      // Loop over pixels in tile to render them
      for (Point2i pixel : tileBounds) {
        // Init pixel
        {
          ProfilePhase pp(Prof::StartPixel);
          tileSampler->StartPixel(pixel);
        }
        // For RNG reproducibility
        if (!InsideExclusive(pixel, pixelBounds))
          continue;

        do {
          // Initialize _CameraSample_ for current sample
          CameraSample cameraSample = tileSampler->GetCameraSample(pixel);
          // Generate camera ray for current sample
          RayDifferential ray;
          Float rayWeight = camera->GenerateRayDifferential(cameraSample, &ray);
          ray.ScaleDifferentials(1 / std::sqrt((Float)tileSampler->samplesPerPixel));
          ++nCameraRays;
          // Evaluate radiance along camera ray and capture Features
          SampleData sf(
            cameraSample.pFilm, // Film position
            cameraSample.pLens, // Lens position
            0.f ,                // L (placeholder)
            rayWeight           // Ray weight
          );
          if (rayWeight > 0) {
            // Evaluate radiance along camera ray
            Li(ray, scene, *tileSampler, arena, sf);
          }
          samplingTile->addSample(pixel, sf);

          // Free _MemoryArena_ memory from computing image sample value
          arena.Reset();

        } while (tileSampler->StartNextSample());
      }
      LOG(INFO) << "Finished sampling tile " << tileBounds;

      // Merge image tile into _SamplingFilm_
      samplingFilm.MergeSamplingTile(std::move(samplingTile));
    }, nTiles);
    reporter_sampling.Done();
  }
  LOG(INFO) << "Sampling finished";
}

  // "Features Mean and StdDev"
  void RPFIntegrator::FillMeanAndStddev(
    const SamplingFilm &samplingFilm,
    SampleFMatrix &pixelFmeanMatrix,
    SampleFMatrix &pixelFstdDevMatrix,
    const int tileSize) 
  {
    // Get bounds
    Bounds2i sampleBounds = camera->film->GetSampleBounds();
    Vector2i sampleExtent = sampleBounds.Diagonal();
    // Divide into tiles
    Point2i nTiles(
      (sampleExtent.x + tileSize - 1) / tileSize,
      (sampleExtent.y + tileSize - 1) / tileSize
    );
    // Init to size
    pixelFmeanMatrix = SampleFMatrix(
      samplingFilm.getWidth(),
      SampleFVector(samplingFilm.getHeight(), SampleF())
    );
    pixelFstdDevMatrix = SampleFMatrix(
      samplingFilm.getWidth(),
      SampleFVector(samplingFilm.getHeight(), SampleF())
    );
    // Loop over pixels
    std::cout << "FillMeanAndStddev" << std::endl;
    ProgressReporter reporter_fstats(nTiles.x * nTiles.y, "Features Mean and StdDev");
    {
      ParallelFor2D([&](Point2i tile) {
        // Compute bounds for tile
        int x0 = sampleBounds.pMin.x + tile.x * tileSize;
        int x1 = std::min(x0 + tileSize, sampleBounds.pMax.x);
        int y0 = sampleBounds.pMin.y + tile.y * tileSize;
        int y1 = std::min(y0 + tileSize, sampleBounds.pMax.y);
        Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));

        // Loop
        for (Point2i pixel : tileBounds) {
          // Get samples for pixel
          SampleDataSet samples = samplingFilm.getPixelSamples(pixel);
          // Get mean and stdDev for each feature
          std::vector<SampleF> vectors;
          for (const SampleData &sd : samples) {
            vectors.push_back(sd.getFeatures());
          }
          pixelFmeanMatrix[pixel.x][pixel.y] = getMean(vectors);
          pixelFstdDevMatrix[pixel.x][pixel.y] = getStdDev(vectors, pixelFmeanMatrix[pixel.x][pixel.y]);
        }
      }, nTiles);
      reporter_fstats.Done();
    }
    LOG(INFO) << "Features Mean and StdDev finished";
  }


  void RPFIntegrator::ComputeCFWeights(
    const SampleDataSet& neighborhood,
    SampleC &Alpha_k,
    SampleF &Beta_k,
    double &W_r_c
  ) {
    // Init dependencies 
    SampleF D_r_fk; // D[r][f,k] = SUM_l MutualInformation(f_k, r_l)
    SampleF D_p_fk; // D[p][f,k] = SUM_l MutualInformation(f_k, p_l)
    SampleC D_r_ck; // D[r][c,k] = SUM_l MutualInformation(c_k, r_l)
    SampleC D_p_ck; // D[p][c,k] = SUM_l MutualInformation(c_k, p_l)
    SampleC D_f_ck; // D[f][c,k] = SUM_l MutualInformation(c_k, f_l)
    // Init to 0
    for (size_t i = 0; i < SD_N_FEATURES; ++i) {
      D_r_fk[i] = 0;
      D_p_fk[i] = 0;
    }
    for (size_t i = 0; i < SD_N_COLOR; ++i) {
      D_r_ck[i] = 0;
      D_p_ck[i] = 0;
      D_f_ck[i] = 0;
    }
    // 1. Aproximate joint mutual information as sum of mutual informations
    // Init data vectors

    std::vector<std::vector<double>> features_data;
    std::vector<std::vector<double>> positions_data;
    std::vector<std::vector<double>> colors_data;
    std::vector<std::vector<double>> random_data;
    for (int i = 0; i < SD_N_FEATURES; ++i) {
      std::vector<double> fi_samples;
      for (const SampleData &sd : neighborhood) {
        fi_samples.push_back(sd.getFeatureI(i));
      }
      features_data.push_back(fi_samples);
    }
    for (int i = 0; i < SD_N_POSITION; ++i) {
      std::vector<double> pi_samples;
      for (const SampleData &sd : neighborhood) {
        pi_samples.push_back(sd.getPositionI(i));
      }
      positions_data.push_back(pi_samples);
    }
    for (int i = 0; i < SD_N_COLOR; ++i) {
      std::vector<double> ci_samples;
      for (const SampleData &sd : neighborhood) {
        ci_samples.push_back(sd.getColorI(i));
      }
      colors_data.push_back(ci_samples);
    }
    for (int i = 0; i < SD_N_RANDOM; ++i) {
      std::vector<double> ri_samples;
      for (const SampleData &sd : neighborhood) {
        ri_samples.push_back(sd.getRandomI(i));
      }
      random_data.push_back(ri_samples);
    }

    // Compute mutual information

    for (int i = 0; i < SD_N_FEATURES; ++i) {
      // For each pair feature x random compute mutual information
      for (int j = 0; j < SD_N_RANDOM; ++j) {
        auto a = features_data[i];
        auto b = random_data[j];
        D_r_fk[i] += MutualInformation(features_data[i], random_data[j]);
      }
      // For each pair feature x position compute mutual information
      for (int j = 0; j < SD_N_POSITION; ++j) {
        D_p_fk[i] += MutualInformation(features_data[i], positions_data[j]);
      }
    }

    for (int i = 0; i < SD_N_COLOR; ++i) {
      // For each pair color x random compute mutual information
      for (int j = 0; j < SD_N_RANDOM; ++j) {
        D_r_ck[i] += MutualInformation(colors_data[i], random_data[j]);
      }
      // For each pair color x position compute mutual information
      for (int j = 0; j < SD_N_POSITION; ++j) {
        D_p_ck[i] += MutualInformation(colors_data[i], positions_data[j]);
      }
      // For each pair color x feature compute mutual information
      for (int j = 0; j < SD_N_FEATURES; ++j) {
        D_f_ck[i] += MutualInformation(colors_data[i], features_data[j]);
      }
    }


    // 2. Dependencies of color x feature, color x position, and feature x position
    // D[f][c] = SUM (D[f][c,k]) for all k
    // D[r][c] = SUM (D[r][c,k]) for all k
    // D[p][c] = SUM (D[p][c,k]) for all k
    double D_f_c=0;
    double D_r_c=0;
    double D_p_c=0;
    for (int i = 0; i < SD_N_COLOR; ++i) {
      D_f_c += D_f_ck[i];
      D_r_c += D_r_ck[i];
      D_p_c += D_p_ck[i];
    }

    // 3. Compute fractional contributions
    // W [c][f,k] = D[c][f,k] / ( D[c][f] + D[c][r] + D[c][p] )
    // W [r][f,k] = D[r][f,k] / ( D[r][f,k] + D[p][f,k] )
    SampleF W_c_fk;
    SampleF W_r_fk;
    for (int i = 0; i < SD_N_FEATURES; ++i) {
      W_c_fk[i] = D_f_ck[i] / (D_f_c + D_r_c + D_p_c);
      W_r_fk[i] = D_r_fk[i] / (D_r_fk[i] + D_p_fk[i]);
    }
    // W [r][c,k] = D[r][c,k] / ( D[r][c] + D[p][c] )
    SampleC W_r_ck;
    for (int i = 0; i < SD_N_COLOR; ++i) {
      W_r_ck[i] = D_r_ck[i] / (D_r_c + D_p_c);
    }
    // 4. Compute Alpha and Beta
    // Alpha_k = 1 - W[r][c,k]
    for (int i = 0; i < SD_N_COLOR; ++i) {
      Alpha_k[i] = 1 - W_r_ck[i];
    }
    // Beta_k = (1 - W[r][f,k]) * W[c][f,k]
    for (int i = 0; i < SD_N_FEATURES; ++i) {
      Beta_k[i] = (1 - W_r_fk[i]) * W_c_fk[i];
    }
    // Compute W_r_c
    W_r_c = D_r_c / (D_r_c + D_p_c);
  }


  // Render
  void RPFIntegrator::Render(const Scene &scene) {  
    // Get bounds
    Bounds2i sampleBounds = camera->film->GetSampleBounds();
    Vector2i sampleExtent = sampleBounds.Diagonal();
    // Divide into tiles
    const int tileSize = 16;
    // Init and Fill SamplingFilm
    std::cout << "Init and Fill SamplingFilm" << std::endl;
    SamplingFilm samplingFilm(sampleBounds);
    FillSampleFilm(
      samplingFilm,
      scene,
      tileSize
    );

    for (size_t i = 0; i < samplingFilm.samples.size(); ++i) {
      for (size_t j = 0; j < samplingFilm.samples[i].size(); ++j) {
        ReportValue(samplesPerPixel, samplingFilm.samples[i][j].size());
      }
    }

    // Write FeatureVector data to file
    std::cout << "Write FeatureVector data to file" << std::endl;
    visualizeSF(
      samplingFilm,
      camera->film->filename
    );



    // PREPROCESSING THE SAMPLES
    // To do this, we compute the average feature vector m{f,p} and the
    // standard deviation vector σf P for each component of the feature
    // vector for the set of samples P at the current pixel. We then cre-
    // ate our neighborhood N using only samples whose features are all
    // within 3 standard deviations of the mean for the pixel

    // 1. Clustering


    // 1.1 Get FEATURES mean and stdDev for each pixel
    SampleFMatrix pixelFmeanMatrix;
    SampleFMatrix pixelFstdDevMatrix;
    FillMeanAndStddev(
      samplingFilm,
      pixelFmeanMatrix,
      pixelFstdDevMatrix,
      tileSize
    );

    // 1.2 Build Neighborhood and 1.3 Normalize
    SamplingFilm neighborhoodFilm(sampleBounds);
  
    int box_size = 3;// 35;
    // Position variances depends on box size
    // (The screen position variance sigma^2_p is set by the filter box size, such that the standard deviation sigma_p is one quarter the width of the box)
    double sigma_p = box_size/4;
    // Input variance for color and f
    double sigma_fc_seed = 0.002;

    // Divide into tiles
    Point2i nTiles(
      (sampleExtent.x + tileSize - 1) / tileSize,
      (sampleExtent.y + tileSize - 1) / tileSize
    );
    ProgressReporter reporter_build_neighborhood(nTiles.x * nTiles.y, "BuildNeighborhood And Normalize");
    {
      ParallelFor2D([&](Point2i tile) {
        // Compute sample bounds for tile
        int x0 = sampleBounds.pMin.x + tile.x * tileSize;
        int x1 = std::min(x0 + tileSize, sampleBounds.pMax.x);
        int y0 = sampleBounds.pMin.y + tile.y * tileSize;
        int y1 = std::min(y0 + tileSize, sampleBounds.pMax.y);
        Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));
        // Get neighborhoodFilm for tile
        std::unique_ptr<SamplingTile> neighborhoodTile = neighborhoodFilm.GetSamplingTile(tileBounds);


        // 1. BUILD NEIGHBORHOOD
        // Loop
        bool debugPrint = true;
        for (Point2i pixel : tileBounds) {
          // Init with pixel samples
          auto neighborhood = samplingFilm.getPixelSamples(pixel);
          // Get surrounding pixels
          auto b_delta = (box_size-1) / 2; // Assumes box_size is odd
          for (int xn = pixel.x - b_delta; xn <= pixel.x + b_delta; ++xn) {
            for (int yn = pixel.y - b_delta; yn <= pixel.y + b_delta; ++yn) {
              // Skip if current pixel
              if (xn == pixel.x && yn == pixel.y) {
                continue;
              }
              // If pixel is outside bounds, skip
              if (xn < sampleBounds.pMin.x || xn >= sampleBounds.pMax.x || yn < sampleBounds.pMin.y || yn >= sampleBounds.pMax.y) {
                continue;
              }
              // Check each sample in pixel
              for (const SampleData &sf : samplingFilm.getPixelSamples( Point2i(xn, yn) )) {
                // Check if all features are within 3 std deviations
                auto sfVec = sf.getFeatures();
                bool within3StdDevs = 
                  allLessThan(
                    absArray(subtractArrays(sfVec, pixelFmeanMatrix[pixel.x][pixel.y])),
                    multiplyArray(pixelFstdDevMatrix[pixel.x][pixel.y], 3) // 3 std devs
                  );
                if (within3StdDevs) {
                  neighborhood.push_back(sf);
                }
              }
            }
          }

          // 2. NORMALIZE NEIGHBORHOOD
          if (neighborhood.size()  == 0) {
            std::cout << "Unexpected empty neighborhood" << std::endl;
            continue;
          }
          // Get mean and stdDev for the full array
          std::vector<SampleX> vectors;
          for (const SampleData &sd : neighborhood) {
            vectors.push_back(sd.getFullArray());
          }
          auto mean = getMean(vectors);
          auto stdDev = getStdDev(vectors, mean);
          // Normalize
          for (auto it = neighborhood.begin(); it != neighborhood.end(); ++it) {
            *it = it->normalized(mean, stdDev);
          }
          // Add samples to neighborhoodTile
          for (const SampleData &sf : neighborhood) {
            neighborhoodTile->addSample(pixel, sf);
          }

          // 3. COMPUTE ALPHA AND BETA FACTORS
          SampleC Alpha_k;
          SampleF Beta_k;
          double W_r_c;
          ComputeCFWeights(
            neighborhood,
            Alpha_k,
            Beta_k,
            W_r_c
          );

          // 4. WEIGHT THE SAMPLES
          // Init sample weights as NxN
          std::vector<std::vector<double>> weights_mat(
            neighborhood.size(),
            std::vector<double>(neighborhood.size(), 0)
          );
          
          // Calculate wij for each pair of samples i,j
          // wij =
          //    exp(- SUM_k ((p_i,k - p_j,k)^2)) / (2 * sigma_p^2)   *
          //    exp(- SUM_k ( ALPHA_k (c_i,k - c_j,k)^2)) / (2 * sigma_n^2)   *
          //    exp(- SUM_k ( BETA_k  (f_i,k - f_j,k)^2)) / (2 * sigma_r^2) 
          for (size_t i = 0; i < neighborhood.size(); ++i) {
            for (size_t j = 0; j < neighborhood.size(); ++j) {
              // Get samples
              auto si = neighborhood[i];
              auto sj = neighborhood[j];
              // Get features
              auto fi = si.getFeatures();
              auto fj = sj.getFeatures();
              // Get colors
              auto ci = si.getColor();
              auto cj = sj.getColor();
              // Get positions
              auto pi = si.getPosition();
              auto pj = sj.getPosition();
              // Calculate wij
              
              // (p_i,k - p_j,k)^2
              auto p_square_diff = squareArray(subtractArrays(pi, pj));
              // (c_i,k - c_j,k)^2
              auto c_square_diff = squareArray(subtractArrays(ci, cj));
              // (f_i,k - f_j,k)^2
              auto f_square_diff = squareArray(subtractArrays(fi, fj));

              // Compute sigmas
              // sigma_f^2 = sigma_c^2 = sigma_fc_seed^2 / (1 - W_r_c)^2
              double sigma_c_squared = sigma_fc_seed * sigma_fc_seed / (1 - W_r_c) / (1 - W_r_c);
              double sigma_f_squared = sigma_c_squared;
              double sigma_p_squared = sigma_p * sigma_p;

              // Calculate wij
              double wij = 
                exp(-sumArray(p_square_diff) / (2 * sigma_p_squared)) *
                exp(-sumArray(multiplyArrays(c_square_diff, Alpha_k)) / (2 * sigma_c_squared)) *
                exp(-sumArray(multiplyArrays(f_square_diff, Beta_k)) / (2 * sigma_f_squared));

              // Save wij
              weights_mat[i][j] = wij;
            }
          }

          // 5. BLEND THE SAMPLES
          
        }
        // Merge neighborhoodTile into neighborhoodFilm
        neighborhoodFilm.MergeSamplingTile(std::move(neighborhoodTile));
      }, nTiles);
      reporter_build_neighborhood.Done();
    }
    LOG(INFO) << "Neighborhood built";
    
    // Get stats of neighborhood sizes
    for (size_t i = 0; i < neighborhoodFilm.samples.size(); ++i) {
      for (size_t j = 0; j < neighborhoodFilm.samples[i].size(); ++j) {
        ReportValue(neighborhoodSize, neighborhoodFilm.samples[i][j].size());
      }
    }


    /*
    // STATISTICAL DEPENDENCY ESTIMATION
    SampleC Alpha_k;
    SampleF Beta_k;
    // Compute
    ComputeCFWeights(
      neighborhood
      Alpha_k,
      Beta_k
    );
    // Print 
    std::cout << "Alpha_k: ";
    for (int i = 0; i < SD_N_COLOR; ++i) {
      std::cout << Alpha_k[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Beta_k: ";
    for (int i = 0; i < SD_N_FEATURES; ++i) {
      std::cout << Beta_k[i] << " ";
    }
    std::cout << std::endl;
    */


    // 1. Mutual Information between Feature k and RandomParameter l using the samples in each Neighborhood

    // 2. Calculate filter weights alpha and beta for each feature
    // > Calculate Dependency of Feature/Color/Position k on All Random Parameters
    // > Calculate Dependency of Color k on All Features
    // > Calculate how all color channels depend on Feature k
    // > Calculate Drc Dpc and Dfc

    // 3. Compute Fractional Contributions
    // Wrfk = Drfk / (Drfk + Dpfk)
    // Wrck = Drck / (Drck + Dpck)
    // Wrc  = 1/3 (Wrc1 + Wrc2 + Wrc3)
    // Wfkc = Dfkc / (Drc + Dpc + Dfc)

    // 4. Compute Filter Weights
    // alpha = 1- Wrck
    // beta = Wfkc ( 1 - Wrfk )

    // FILTERING THE SAMPLES
    // > calc wij
    // > Blend samples









    // Render
    // Get filmTile
    std::unique_ptr<FilmTile> filmTile = camera->film->GetFilmTile(sampleBounds);
    auto samples = samplingFilm.samples;
    // Add camera ray's contribution to image
    for (int x = 0; x < sampleExtent.x; ++x) {
      for (int y = 0; y < sampleExtent.y; ++y) {
        for (const SampleData &sf : samples[x][y]) {
          filmTile->AddSample(sf.getPFilm(), sf.getL(), sf.rayWeight); // AddSplat instead?
        }
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
    SampleData &sf,
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
        sf.setN0(isect.n);
        sf.setP0(isect.p);
      } else if (bounces == 1) {
        sf.setN1(isect.n);
        sf.setP1(isect.p);
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
  //sf.L = L;
  sf.setL(L);
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