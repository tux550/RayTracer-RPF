#include "visualization/vis.h"

namespace pbrt {

std::vector<std::vector<BasicRGB>> createBasicRGBMatrix(
  size_t nRows,
  size_t nCols
) {
  return std::vector<std::vector<BasicRGB>>(
    nRows,
    std::vector<BasicRGB>(nCols, {0, 0, 0})
  );
}

  void writeRGBMatrix(
    const std::vector<std::vector<BasicRGB>> &rgbMatrix,
    const std::string &filename
  ) {
    size_t nRows = rgbMatrix.size();
    size_t nCols = rgbMatrix[0].size();
    Imf::Rgba* pixels = new Imf::Rgba[nCols * nRows];
    for (size_t i = 0; i < nRows; ++i) {
      for (size_t j = 0; j < nCols; ++j) {
        pixels[j * nRows + i] = Imf::Rgba(rgbMatrix[i][j].r, rgbMatrix[i][j].g, rgbMatrix[i][j].b, 1);
      }
    }
    Imf::RgbaOutputFile file(filename.c_str(), nCols, nRows, Imf::WRITE_RGBA);
    file.setFrameBuffer(pixels, 1, nCols);
    file.writePixels(nRows);
    delete[] pixels;
  }

  void normalizeRGBMatrix (
    std::vector<std::vector<BasicRGB>> &rgbMatrix
  ) {
    // Find max values
    BasicRGB maxRGB = {0, 0, 0};
    for (size_t i = 0; i < rgbMatrix.size(); ++i) {
      for (size_t j = 0; j < rgbMatrix[0].size(); ++j) {
        maxRGB.r = std::max(maxRGB.r, rgbMatrix[i][j].r);
        maxRGB.g = std::max(maxRGB.g, rgbMatrix[i][j].g);
        maxRGB.b = std::max(maxRGB.b, rgbMatrix[i][j].b);
      }
    }
    // Normalize
    for (size_t i = 0; i < rgbMatrix.size(); ++i) {
      for (size_t j = 0; j < rgbMatrix[0].size(); ++j) {
        rgbMatrix[i][j] = rgbMatrix[i][j] / maxRGB;
      }
    }
  }

} // namespace pbrt
