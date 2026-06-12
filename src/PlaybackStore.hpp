#pragma once

// PlaybackStore - per-level library of finished training runs. Every
// completed (or cancelled-with-progress) training session appends its best
// run's action tape here, so a session that finishes unattended is never
// lost: the playback can be re-watched any time from the settings popup.
//
// One .dat file per level under <mod save dir>/playbacks/, deliberately a
// plain user-accessible file so playbacks can be backed up or shared.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

class GJGameLevel;

namespace neatgd {

// One saved run: the open-loop action tape (see Genome::tapeToggles) plus
// display metadata. Tape-only - no network - so replaying past reachStep
// just keeps the button released; the tape IS the whole recorded run.
struct Playback {
    std::string name;      // user-editable, defaults to a timestamp
    int64_t timestamp = 0; // unix seconds, when the run was saved
    double percent = 0.0;  // best fitness of the run
    int reachStep = 0;     // decision step the run ended on
    std::vector<int> toggles;
};

namespace PlaybackStore {

// Stable per-level key: the online level id, or the (sanitized) level name
// for local/editor levels that have no global id.
std::string levelKeyFor(GJGameLevel* level);

std::filesystem::path fileFor(std::string const& levelKey);

// Missing/corrupt files load as an empty list (never throws).
std::vector<Playback> load(std::string const& levelKey);
bool save(std::string const& levelKey, std::vector<Playback> const& list);
bool append(std::string const& levelKey, Playback playback);

} // namespace PlaybackStore

} // namespace neatgd
