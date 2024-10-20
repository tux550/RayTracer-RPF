
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
  STAT_PERCENT("Integrator/Zero-radiance paths", zeroRadiancePaths, totalPaths);
  STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);

  void writeFVMat(const std::vector<std::vector<
 std::vector<SampleData>>> &fvMat, const std::string &filename) {
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

    size_t nRows = fvMat.size();
    size_t nCols = fvMat[0].size();
    // Calculate magnitude of normal vectors and position vectors
    std::vector<std::vector<BasicRGB>> n0Mag(nRows, std::vector<BasicRGB>(nCols, {0,0,0}));
    std::vector<std::vector<BasicRGB>> n1Mag(nRows, std::vector<BasicRGB>(nCols, {0,0,0}));
    std::vector<std::vector<BasicRGB>> p0Mag(nRows, std::vector<BasicRGB>(nCols, {0,0,0}));
    std::vector<std::vector<BasicRGB>> p1Mag(nRows, std::vector<BasicRGB>(nCols, {0,0,0}));
    // Calculate Film and Lens position
    std::vector<std::vector<BasicRGB>> filmPos(nRows, std::vector<BasicRGB>(nCols, {0,0,0}));
    std::vector<std::vector<BasicRGB>> lensPos(nRows, std::vector<BasicRGB>(nCols, {0,0,0}));
    
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        for (const SampleData &sd : fvMat[i][j]) {
          n0Mag[i][j] = n0Mag[i][j] + BasicRGB(sd.fv.n0.x, sd.fv.n0.y, sd.fv.n0.z);
          n1Mag[i][j] = n1Mag[i][j] + BasicRGB(sd.fv.n1.x, sd.fv.n1.y, sd.fv.n1.z);
          p0Mag[i][j] = p0Mag[i][j] + BasicRGB(sd.fv.p0.x, sd.fv.p0.y, sd.fv.p0.z);
          p1Mag[i][j] = p1Mag[i][j] + BasicRGB(sd.fv.p1.x, sd.fv.p1.y, sd.fv.p1.z);
          filmPos[i][j] = filmPos[i][j] + BasicRGB(sd.pFilm.x, sd.pFilm.y, 0);
          lensPos[i][j] = lensPos[i][j] + BasicRGB(sd.pLens.x, sd.pLens.y, 0);
        }
        n0Mag[i][j] = n0Mag[i][j] / fvMat[i][j].size();
        n1Mag[i][j] = n1Mag[i][j] / fvMat[i][j].size();
        p0Mag[i][j] = p0Mag[i][j] / fvMat[i][j].size();
        p1Mag[i][j] = p1Mag[i][j] / fvMat[i][j].size();
        filmPos[i][j] = filmPos[i][j] / fvMat[i][j].size();
        lensPos[i][j] = lensPos[i][j] / fvMat[i][j].size();
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
        maxN0.r = std::max(maxN0.r, n0Mag[i][j].r);
        maxN0.g = std::max(maxN0.g, n0Mag[i][j].g);
        maxN0.b = std::max(maxN0.b, n0Mag[i][j].b);
        maxN1.r = std::max(maxN1.r, n1Mag[i][j].r);
        maxN1.g = std::max(maxN1.g, n1Mag[i][j].g);
        maxN1.b = std::max(maxN1.b, n1Mag[i][j].b);
        maxP0.r = std::max(maxP0.r, p0Mag[i][j].r);
        maxP0.g = std::max(maxP0.g, p0Mag[i][j].g);
        maxP0.b = std::max(maxP0.b, p0Mag[i][j].b);
        maxP1.r = std::max(maxP1.r, p1Mag[i][j].r);
        maxP1.g = std::max(maxP1.g, p1Mag[i][j].g);
        maxP1.b = std::max(maxP1.b, p1Mag[i][j].b);
        maxFilmPos.r = std::max(maxFilmPos.r, filmPos[i][j].r);
        maxFilmPos.g = std::max(maxFilmPos.g, filmPos[i][j].g);
        maxLensPos.r = std::max(maxLensPos.r, lensPos[i][j].r);
        maxLensPos.g = std::max(maxLensPos.g, lensPos[i][j].g);
      }
    }
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        n0Mag[i][j] = (n0Mag[i][j] / maxN0) * 255;
        n1Mag[i][j] = (n1Mag[i][j] / maxN1) * 255;
        p0Mag[i][j] = (p0Mag[i][j] / maxP0) * 255;
        p1Mag[i][j] = (p1Mag[i][j] / maxP1) * 255;
        filmPos[i][j] = (filmPos[i][j] / maxFilmPos) * 255;
        lensPos[i][j] = (lensPos[i][j] / maxLensPos) * 255;
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
        n0pixels[j * nRows + i] = Imf::Rgba(n0Mag[i][j].r, n0Mag[i][j].g, n0Mag[i][j].b, 1);
        n1pixels[j * nRows + i] = Imf::Rgba(n1Mag[i][j].r, n1Mag[i][j].g, n1Mag[i][j].b, 1);
        p0pixels[j * nRows + i] = Imf::Rgba(p0Mag[i][j].r, p0Mag[i][j].g, p0Mag[i][j].b, 1);
        p1pixels[j * nRows + i] = Imf::Rgba(p1Mag[i][j].r, p1Mag[i][j].g, p1Mag[i][j].b, 1);
        // Film and Lens position
        filmPosPixels[j * nRows + i] = Imf::Rgba(filmPos[i][j].r, filmPos[i][j].g, 0, 1);
        lensPosPixels[j * nRows + i] = Imf::Rgba(lensPos[i][j].r, lensPos[i][j].g, 0, 1);

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

  // Render
  void RPFIntegrator::Render(const Scene &scene) {  
    Bounds2i sampleBounds = camera->film->GetSampleBounds();
    Vector2i sampleExtent = sampleBounds.Diagonal();
    ProgressReporter reporter(sampleExtent.x*sampleExtent.y, "Rendering");
    // Allocate 3D matrix of samples
    std::vector<std::vector<std::vector<SampleData>>> samples(
      sampleExtent.x,
      std::vector<std::vector<SampleData>>(
        sampleExtent.y,
        std::vector<SampleData>()
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
          // Evaluate radiance along camera ray and capture FeatureVector
          Spectrum L(0.f);
          FeatureVector fv;
          if (rayWeight > 0) {
            L = Li(ray, scene, *sampler, arena, fv);
          }
          // Save sample data
          SampleData sd(
            cameraSample.pFilm,
            L,
            rayWeight,
            fv
          );
          samples[pixel.x][pixel.y].push_back(sd);
          // Free _MemoryArena_ memory from computing image sample value
          arena.Reset();
        } while (sampler->StartNextSample());        
      }
    }
    LOG(INFO) << "Finished sampling pixels" << sampleBounds;
    // Write FeatureVector data to file
    writeFVMat(samples, camera->film->filename);
    // Get filmTile
    std::unique_ptr<FilmTile> filmTile = camera->film->GetFilmTile(sampleBounds);
    // Add camera ray's contribution to image
    for (int x = 0; x < sampleExtent.x; ++x) {
      for (int y = 0; y < sampleExtent.y; ++y) {
        for (const SampleData &sd : samples[x][y]) {
          filmTile->AddSample(sd.pFilm, sd.L, sd.rayWeight);
        }
      }
    }
    // Merge image tile into _Film_
    camera->film->MergeFilmTile(std::move(filmTile));
    LOG(INFO) << "Rendering finished";

    // Save final image after rendering
    camera->film->WriteImage();
  }

  // Luminance samplings
  Spectrum RPFIntegrator::Li(
    const RayDifferential &r, const Scene &scene,
    Sampler &sampler, MemoryArena &arena,
    FeatureVector &fv,
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
        fv.n0 = isect.n;
        fv.p0 = isect.p;
      } else if (bounces == 1) {
        fv.n1 = isect.n;
        fv.p1 = isect.p;
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

  return L;
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