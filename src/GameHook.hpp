#pragma once

#include <cstddef>
#include <vector>

class PlayLayer;

namespace neatgd {

constexpr size_t SCAN_SAMPLES = 8;
constexpr float SCAN_OFFSETS[SCAN_SAMPLES] = {
    0.f, 40.f, 80.f, 120.f, 160.f, 200.f, 250.f, 300.f};

constexpr size_t INPUT_COUNT = 12 + 4 * SCAN_SAMPLES;

constexpr float NORM_Y = 300.f;
constexpr float NORM_VEL = 16.f;
constexpr float NORM_DY = 120.f;
constexpr float NORM_CLEAR = 240.f;

struct LevelObject {
    float x = 0.f;
    float y = 0.f;
    float w = 30.f;
    float h = 30.f;
    bool hazard = false;
    bool slope = false;
    int objectId = 0;
};

std::vector<LevelObject> extractLevelObjects(PlayLayer* layer);

}
