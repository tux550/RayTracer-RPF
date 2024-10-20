
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
  STAT_PERCENT("Integrator/Zero-radiance paths", zeroRadiancePaths, totalPaths);
  STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);

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
    /*
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
          SampleData sd;
          sd.pFilm = cameraSample.pFilm;
          sd.L = L;
          sd.rayWeight = rayWeight;
          sd.fv = fv;
          samples[pixel.x][pixel.y].push_back(sd);
          // Free _MemoryArena_ memory from computing image sample value
          arena.Reset();
        } while (tileSampler->StartNextSample());        
      }
    }
    LOG(INFO) << "Finished sampling pixels" << sampleBounds;
    // Add camera ray's contribution to image
    for (int x = 0; x < sampleExtent.x; ++x) {
      for (int y = 0; y < sampleExtent.y; ++y) {
        for (const SampleData &sd : samples[x][y]) {
          camera->film->AddSample(sd.pFilm, sd.L, sd.rayWeight);
        }
      }
    }
    LOG(INFO) << "Rendering finished";
    */
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
    /*    
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
  */
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