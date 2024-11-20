#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_ARRAY_OPS_H
#define PBRT_ARRAY_OPS_H

#include <cmath>
#include <vector>
#include <array>

namespace pbrt {


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

// Square of an array
template <typename T, size_t N>
std::array<T,N> squareArray(const std::array<T,N> &a) {
  std::array<T,N> result;
  for (size_t i = 0; i < N; ++i) {
    result[i] = a[i] * a[i];
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



}


#endif  // PBRT_ARRAY_OPS_H