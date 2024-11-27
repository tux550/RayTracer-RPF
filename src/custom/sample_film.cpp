#include "custom/sample_film.h"

namespace pbrt {

SamplingTile::SamplingTile(const Bounds2i &pixelBounds)
    : pixelBounds(pixelBounds) {
    samples = std::vector<std::vector<SampleDataSet>>(
        pixelBounds.Diagonal().x,
        std::vector<SampleDataSet>(pixelBounds.Diagonal().y, SampleDataSet()));
}

void SamplingTile::addSample(const Point2i &pixel, const SampleData &sample) {
    // Get adjusted x and y coordinates
    auto adjustedX = pixel.x - pixelBounds.pMin.x;
    auto adjustedY = pixel.y - pixelBounds.pMin.y;
    // Add sample to the list of samples
    samples[adjustedX][adjustedY].push_back(sample);
}

SampleDataSet SamplingTile::getPixelSamples(const Point2i &pixel) {
    // Get adjusted x and y coordinates
    auto adjustedX = pixel.x - pixelBounds.pMin.x;
    auto adjustedY = pixel.y - pixelBounds.pMin.y;
    return samples[adjustedX][adjustedY];
}

SamplingFilm::SamplingFilm(const Bounds2i &pixelBounds)
    : pixelBounds(pixelBounds), samples(), mutex() {
    samples = std::vector<std::vector<SampleDataSet>>(
        pixelBounds.Diagonal().x,
        std::vector<SampleDataSet>(pixelBounds.Diagonal().y, SampleDataSet()));
}

void SamplingFilm::AddSample(const Point2i &pixel, const SampleData &sample) {
    // Get adjusted x and y coordinates
    auto adjustedX = pixel.x - pixelBounds.pMin.x;
    auto adjustedY = pixel.y - pixelBounds.pMin.y;
    samples[adjustedX][adjustedY].push_back(sample);
}

std::unique_ptr<SamplingTile> SamplingFilm::GetSamplingTile(
    const Bounds2i &sampleBounds) {
    Bounds2i tilePixelBounds = Intersect(sampleBounds, pixelBounds);
    return std::unique_ptr<SamplingTile>(new SamplingTile(tilePixelBounds));
}

void SamplingFilm::MergeSamplingTile(std::unique_ptr<SamplingTile> tile) {
    // Thread-safe merge of tile samples into film samples
    std::lock_guard<std::mutex> lock(mutex);
    for (Point2i pixel : tile->pixelBounds) {
        SampleDataSet tileSamples = tile->getPixelSamples(pixel);
        // Add samples to tile
        for (const SampleData &sample : tileSamples) {
            AddSample(pixel, sample);
        }
    }
}

int SamplingFilm::getWidth() const { return pixelBounds.Diagonal().x; }

int SamplingFilm::getHeight() const { return pixelBounds.Diagonal().y; }

SampleDataSet SamplingFilm::getPixelSamples(const Point2i &pixel) const {
    // Get adjusted x and y coordinates
    auto adjustedX = pixel.x - pixelBounds.pMin.x;
    auto adjustedY = pixel.y - pixelBounds.pMin.y;
    return samples[adjustedX][adjustedY];
}

SampleData SamplingFilm::getPixelSampleI(const Point2i &pixel, int i) const {
    auto adjustedX = pixel.x - pixelBounds.pMin.x;
    auto adjustedY = pixel.y - pixelBounds.pMin.y;
    return samples[adjustedX][adjustedY][i];
}
}  // namespace pbrt
