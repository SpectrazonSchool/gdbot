#include "PlaybackStore.hpp"

#include <Geode/Geode.hpp>

#include <cctype>
#include <fstream>

using namespace geode::prelude;

namespace neatgd {

namespace {

constexpr uint32_t MAGIC = 0x5044474E;
constexpr uint32_t VERSION = 1;
constexpr uint32_t MAX_ENTRIES = 10000;
constexpr uint32_t MAX_NAME = 1024;
constexpr uint32_t MAX_TOGGLES = 10'000'000;

static_assert(sizeof(int) == 4, "tape toggles are serialized as i32");

template <typename T>
bool readPod(std::istream& in, T& out) {
    in.read(reinterpret_cast<char*>(&out), sizeof(T));
    return in.good();
}

template <typename T>
void writePod(std::ostream& out, T const& value) {
    out.write(reinterpret_cast<char const*>(&value), sizeof(T));
}

}

namespace PlaybackStore {

std::string levelKeyFor(GJGameLevel* level) {
    if (!level) return "unknown";
    if (int const id = level->m_levelID.value(); id > 0) {
        return fmt::format("{}", id);
    }
    std::string name = level->m_levelName;
    for (auto& c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
    }
    return name.empty() ? "unnamed" : "local_" + name;
}

std::filesystem::path fileFor(std::string const& levelKey) {
    return Mod::get()->getSaveDir() / "playbacks" / (levelKey + ".dat");
}

std::vector<Playback> load(std::string const& levelKey) {
    std::vector<Playback> list;
    std::ifstream in(fileFor(levelKey), std::ios::binary);
    if (!in) return list;

    uint32_t magic = 0, version = 0, count = 0;
    if (!readPod(in, magic) || magic != MAGIC) return list;
    if (!readPod(in, version) || version != VERSION) return list;
    if (!readPod(in, count) || count > MAX_ENTRIES) return list;

    list.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        Playback p;
        uint32_t nameLen = 0;
        if (!readPod(in, nameLen) || nameLen > MAX_NAME) break;
        p.name.resize(nameLen);
        if (nameLen > 0) in.read(p.name.data(), nameLen);
        uint32_t nToggles = 0;
        if (!readPod(in, p.timestamp) || !readPod(in, p.percent)
            || !readPod(in, p.reachStep) || !readPod(in, nToggles)
            || nToggles > MAX_TOGGLES) {
            break;
        }
        p.toggles.resize(nToggles);
        if (nToggles > 0) {
            in.read(reinterpret_cast<char*>(p.toggles.data()),
                    static_cast<std::streamsize>(nToggles) * sizeof(int));
        }
        if (!in) break;
        list.push_back(std::move(p));
    }
    return list;
}

bool save(std::string const& levelKey, std::vector<Playback> const& list) {
    auto const path = fileFor(levelKey);
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        log::warn("NEATGD: cannot write playback file {}", path.string());
        return false;
    }
    writePod(out, MAGIC);
    writePod(out, VERSION);
    writePod(out, static_cast<uint32_t>(list.size()));
    for (auto const& p : list) {
        writePod(out, static_cast<uint32_t>(p.name.size()));
        if (!p.name.empty()) out.write(p.name.data(), p.name.size());
        writePod(out, p.timestamp);
        writePod(out, p.percent);
        writePod(out, p.reachStep);
        writePod(out, static_cast<uint32_t>(p.toggles.size()));
        if (!p.toggles.empty()) {
            out.write(reinterpret_cast<char const*>(p.toggles.data()),
                      static_cast<std::streamsize>(p.toggles.size())
                          * sizeof(int));
        }
    }
    return out.good();
}

bool append(std::string const& levelKey, Playback playback) {
    auto list = load(levelKey);
    list.push_back(std::move(playback));
    return save(levelKey, list);
}

}

}
