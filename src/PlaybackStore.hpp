#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

class GJGameLevel;

namespace neatgd {

struct Playback {
    std::string name;
    int64_t timestamp = 0;
    double percent = 0.0;
    int reachStep = 0;
    std::vector<int> toggles;
    bool completed = true;
};

namespace PlaybackStore {

std::string levelKeyFor(GJGameLevel* level);

std::filesystem::path fileFor(std::string const& levelKey);
std::filesystem::path sessionFileFor(std::string const& levelKey, int64_t id);
bool sessionExists(std::string const& levelKey, int64_t id);
bool deleteSession(std::string const& levelKey, int64_t id);

std::vector<Playback> load(std::string const& levelKey);
bool save(std::string const& levelKey, std::vector<Playback> const& list);
bool append(std::string const& levelKey, Playback playback);

}

}
