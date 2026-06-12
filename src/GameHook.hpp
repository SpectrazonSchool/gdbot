#pragma once

// GameHook - the PlayLayer side of the mod. NEAT genomes are evaluated
// *live*: the real engine runs (fast-forwarded via the scheduler time scale
// during training), the hook reads true game state each frame, feeds it to
// the active network and injects the jump button. Fitness is the game's own
// getCurrentPercent(), so there is no simulation to desync from.
//
// The $modify class lives in GameHook.cpp; this header defines the network
// input contract and the level-geometry snapshot used to build those inputs.

#include <cstddef>
#include <vector>

class PlayLayer;

namespace neatgd {

// Terrain scan: the level is sampled at fixed X offsets ahead of the player
// (denser close by, where timing is decided). Per sample column the network
// sees the corridor: floor height, ceiling height, and the nearest hazard
// above and below its own Y. This generalizes across gamemodes - a cube
// reads the floor line for jumps, a ship reads floor + ceiling + hazards to
// thread gaps (fly under walls, hold the middle of spike corridors).
constexpr size_t SCAN_SAMPLES = 8;
constexpr float SCAN_OFFSETS[SCAN_SAMPLES] = {
    0.f, 40.f, 80.f, 120.f, 160.f, 200.f, 250.f, 300.f};

// Network input layout, in order (all roughly normalized to [-1, 1]):
//   [0]   player Y (relative to ground level)
//   [1]   player Y velocity
//   [2]   on ground (0/1) - i.e. a jump is available
//   [3]   jump button currently held (0/1)
//   [4]   gravity flipped (0/1)
//   [5..11] gamemode one-hot: ship, ufo, ball, wave, robot, spider, swing
//           (all zero = cube)
//   then, per scan column (S = SCAN_SAMPLES). The scan is encoded so that a
//   clear path reads as 0 on every column - "nothing ahead" is the resting
//   state - which is what lets the network sit still through empty stretches:
//   [12 .. 12+S-1]    floor: standable solid top relative to the player's
//                     feet (0 = flat ground, + = ledge above, - = drop)
//   [12+S .. 12+2S-1] ceiling proximity above the head (0 = clear,
//                     1 = blocked at head height)
//   [12+2S .. ]       hazard proximity above player Y (0 = none/far,
//                     1 = at player Y)
//   [12+3S .. ]       hazard proximity below player Y (0 = none/far,
//                     1 = at player Y)
constexpr size_t INPUT_COUNT = 12 + 4 * SCAN_SAMPLES;

// Normalization scales (consistency matters more than exactness for NEAT).
constexpr float NORM_Y = 300.f;
// m_yVelocity is stored in units per 1/60s tick, NOT units/second: a cube
// jump starts at ~ +11.2 and terminal fall is ~ -15. (Was 600 - which
// squashed the velocity input to ~zero and left the network nearly blind
// to its own vertical momentum.)
constexpr float NORM_VEL = 16.f;
constexpr float NORM_DY = 120.f;    // floor offsets, ~4 blocks
constexpr float NORM_CLEAR = 240.f; // ceiling/hazard clearances, ~8 blocks

// One collidable object, snapshot of its real collision rect
// (GameObject::getObjectRect) taken once per level load.
struct LevelObject {
    float x = 0.f; // hitbox center, GD units
    float y = 0.f;
    float w = 30.f;
    float h = 30.f;
    bool hazard = false; // hazard (spike/saw) vs solid (block/slope)
    bool slope = false;
    int objectId = 0; // GD object ID, diagnostics only
};

// Snapshot the level's collidable geometry, sorted by X. Main thread only.
std::vector<LevelObject> extractLevelObjects(PlayLayer* layer);

} // namespace neatgd
