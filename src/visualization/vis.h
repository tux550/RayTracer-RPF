#include <ImfRgbaFile.h>
#include <ImfArray.h>
#include <vector>


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

  // Write RGB Matrix to file
  void writeRGBMatrix(
    const std::vector<std::vector<BasicRGB>> &rgbMatrix,
    const std::string &filename
  ) ;

  // Normalize RGB Matrix
  void normalizeRGBMatrix (
    std::vector<std::vector<BasicRGB>> &rgbMatrix
  ); 



}