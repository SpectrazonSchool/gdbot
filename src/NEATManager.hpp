#pragma once

#include "Neat.hpp"
#include "PlaybackStore.hpp"

#include <chrono>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace neatgd {

// User-tunable training parameters, set from the settings popup. Values are
// already clamped to sane ranges by the popup before they get here.
struct TrainingConfig {
    int population = 50;
    int maxGenerations = 50;
    // When the all-time best hasn't improved for this many generations, the
    // frontier is assumed to be an impossible position (an input that gained
    // distance but cornered the player) and the locked-in replay is rolled
    // back - see backtrackFrontier. Training itself only stops on the
    // generation/time limits, or once a rollback reaches the level start.
    int stagnationLimit = 15;
    // Wall-clock budget in minutes; 0 = unlimited. Checked between
    // generations.
    double maxMinutes = 10.0;
    // Game speed multiplier while training (applied by the
    // CCScheduler::update hook). Physics are unaffected - GD quantizes dt
    // into fixed-rate substeps - only wall-clock time shrinks. The default
    // is higher than the old batch system's 8x: with a single player
    // instead of a ghost wave the engine has headroom to spare.
    double speed = 16.0;
    // Frame-rate target while training (GameManager::m_customFPSTarget),
    // restored to the player's setting when training ends. The Speed
    // multiplier only takes real effect if each frame's scaled delta stays
    // under the engine's per-frame physics-substep cap; past that the surplus
    // just piles into GJBaseGameLayer::m_extraDelta and the nominal speed is
    // never delivered. Running more (cheap, with graphics hidden) frames per
    // second raises that ceiling roughly linearly. Capped by vsync / the
    // monitor refresh if the player has vsync on.
    double maxFps = 360.0;
    // Hide level rendering while training (status HUD over black), freeing
    // hardware for higher speed values.
    bool hideGraphics = false;
};

// NEATManager - orchestrates live sequential training. Genomes are evaluated
// one per attempt by driving the real player through the full, unmodified
// engine - exactly the code path the showcase uses - so a genome's training
// fitness is the same physical quantity its showcase run reproduces.
//
// Frontier exploration: every attempt starts by replaying the genome's
// inherited action tape (its parent's proven button history - see
// Genome::tapeToggles) and hands control to the network only shortly before
// the parent's death point. Mutation therefore expresses 0% at the start of
// an attempt and 100% at its end: sections already passed can't be lost to
// a bad mutation, and all exploration concentrates at the frontier.
//
// Backtracking: a frontier that maximizes distance can still be a trap - an
// input that gained ground but left the player in an unwinnable position.
// Stagnation is how that shows up, so instead of ending training it rolls
// the replay cap back a handful of presses (further each round) and the
// population re-explores from before the trap. Any new all-time best lifts
// the cap: the trap was escaped, normal frontier ratcheting resumes.
//
//   resetLevel       -> endAttempt() for an unfinished attempt, then
//                       beginAttempt() for the next genome
//   each substep     -> shouldHoldCurrent(inputs, holding)
//   player death     -> endAttempt(percent, died=true, jumps)
//   level end        -> endAttempt(100, died=false, jumps)
//
// Phases: Idle -> Training -> Idle + finish prompt (training often ends
// unattended, so instead of auto-playing the one-shot showcase the PlayLayer
// hook pauses the game and asks) -> Showcase (best genome replays on the
// real player once at 1x speed) -> Idle. The best run is also saved to the
// level's playback library (PlaybackStore) the moment training ends, so
// declining the prompt loses nothing - saved playbacks replay through the
// same Showcase phase, tape-only.
class NEATManager {
public:
    enum class Phase { Idle, Training, Showcase };

    static NEATManager* get();

    Phase phase() const { return m_phase; }
    bool isActive() const { return m_phase != Phase::Idle; }
    // Speed multiplier the CCScheduler::update hook applies while training.
    double trainingSpeed() const { return m_config.speed; }
    bool hideGraphics() const { return m_config.hideGraphics; }

    bool beginTraining(TrainingConfig config);
    void stop(char const* reason);

    // --- Training attempts ------------------------------------------------
    // Builds the network for the next genome to evaluate; returns false if
    // not training.
    bool beginAttempt();
    bool attemptInProgress() const { return m_attemptActive; }
    // True while the current decision step falls inside the replayed tape
    // prefix - the caller can skip building network inputs entirely, the
    // tape decides alone.
    bool replaying() const {
        return (m_attemptActive || m_showcaseActive)
            && m_step < m_takeoverStep;
    }
    // `holding` is the player's current button state - the output threshold
    // has hysteresis (press above 0.55, release below 0.45) so a wavering
    // output doesn't jitter the button at substep rate.
    bool shouldHoldCurrent(std::vector<double> const& inputs, bool holding);
    // Records the current genome's result and advances to the next one; at a
    // generation boundary runs the stop checks and epoch. May change phase!
    // `jumps` = button presses made during the run; a small fitness penalty
    // per press makes jump spam strictly worse than clean play at equal
    // distance. Best-genome tracking and the HUD still use the raw percent.
    void endAttempt(double percent, bool died, int jumps);

    // --- Showcase ---------------------------------------------------------
    // One-shot fetch of the end-of-training prompt text; true while a prompt
    // is waiting to be shown. Consumed by the PlayLayer hook, which pauses
    // the game and shows the watch-now popup.
    bool takeFinishedPrompt(std::string& textOut);
    // "Watch now" on the finish prompt: re-arms the showcase for the genome
    // this session just trained. The caller restarts the level.
    bool armShowcase();
    // Arms a tape-only showcase of a saved playback from the level library.
    // The caller restarts the level. Fails unless idle.
    bool beginPlayback(Playback const& playback);
    void onShowcaseStart();
    bool showcaseActive() const { return m_showcaseActive; }
    bool shouldHold(std::vector<double> const& inputs, bool holding);
    void onShowcaseEnd(double percent);

    // One-line status for the in-game HUD label.
    std::string statusText() const;
    // Multi-line training progress + ETA for the hidden-graphics screen.
    std::string progressText() const;

private:
    void finishTraining(bool solved, char const* reason);
    // Append the session's best run to the current level's playback library
    // (no-op without progress or a PlayLayer). Called when training ends -
    // including cancellation - so an unattended run is never lost.
    void savePlayback();
    // Raise the game's frame-rate target to m_config.maxFps for the duration
    // of training (so the Speed multiplier is actually delivered rather than
    // capped by the engine's per-frame substep limit), and put it back when
    // training ends.
    void applyTrainingFps();
    void restoreFps();
    // Tape state at decision step `step`; cursor-based, so steps must be
    // queried in increasing order within an attempt.
    bool tapeStateAt(int step);
    // Rolls m_frontierCap back BACKTRACK_INPUTS presses within the best
    // tape (counting from the current cap), so replay stops before the
    // suspected trap and the network re-decides from there.
    void backtrackFrontier();

    Phase m_phase = Phase::Idle;
    TrainingConfig m_config;
    std::unique_ptr<Population> m_population;

    // Current attempt.
    std::unique_ptr<Network> m_currentNet;
    bool m_attemptActive = false;
    int m_cursor = 0; // genome being (or next to be) evaluated

    // Tape replay/recording for the current attempt (or showcase run).
    std::vector<int> m_replayToggles; // inherited tape being replayed
    size_t m_replayCursor = 0;
    bool m_replayHold = false;
    int m_takeoverStep = 0; // first step the network controls
    int m_step = 0;         // decision steps since attempt start
    std::vector<int> m_recordToggles; // this attempt's actual history
    bool m_recordHold = false;
    // Backtracking cap on every attempt's takeover step: no tape is
    // replayed past it. INT_MAX = uncapped (normal frontier ratcheting).
    int m_frontierCap = std::numeric_limits<int>::max();
    std::mt19937 m_rng{std::random_device{}()};

    int m_generation = 0;
    int m_sinceImproved = 0;
    double m_bestFitness = 0.0;
    double m_generationBest = 0.0;
    bool m_solved = false;
    Genome m_bestGenome;
    std::chrono::steady_clock::time_point m_trainStart;

    std::unique_ptr<Network> m_showcaseNet;
    bool m_showcaseActive = false;

    // End-of-training prompt, pending until the PlayLayer hook collects it.
    bool m_promptPending = false;
    std::string m_promptText;

    // Player's frame-rate target, saved while training overrides it.
    float m_savedFpsTarget = 0.f;
    bool m_fpsOverridden = false;
};

} // namespace neatgd
