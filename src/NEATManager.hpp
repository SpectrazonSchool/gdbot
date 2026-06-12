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

struct TrainingConfig {
    int population = 50;
    int maxGenerations = 50;
    int stagnationLimit = 15;
    double maxMinutes = 10.0;
    double speed = 16.0;
    double maxFps = 360.0;
    bool hideGraphics = false;
};

class NEATManager {
public:
    enum class Phase { Idle, Training, Showcase };

    static NEATManager* get();

    Phase phase() const { return m_phase; }
    bool isActive() const { return m_phase != Phase::Idle; }
    double trainingSpeed() const { return m_config.speed; }
    bool hideGraphics() const { return m_config.hideGraphics; }

    bool beginTraining(TrainingConfig config);
    void stop(char const* reason);

    bool beginAttempt();
    bool attemptInProgress() const { return m_attemptActive; }
    bool replaying() const {
        return (m_attemptActive || m_showcaseActive)
            && m_step < m_takeoverStep;
    }
    bool shouldHoldCurrent(std::vector<double> const& inputs, bool holding);
    void endAttempt(double percent, bool died, int jumps);

    bool takeFinishedPrompt(std::string& textOut);
    bool armShowcase();
    bool beginPlayback(Playback const& playback);
    void onShowcaseStart();
    bool showcaseActive() const { return m_showcaseActive; }
    bool shouldHold(std::vector<double> const& inputs, bool holding);
    void onShowcaseEnd(double percent);

    std::string statusText() const;
    std::string progressText() const;

private:
    void finishTraining(bool solved, char const* reason);
    void savePlayback();
    void applyTrainingFps();
    void restoreFps();
    bool tapeStateAt(int step);
    void backtrackFrontier();

    Phase m_phase = Phase::Idle;
    TrainingConfig m_config;
    std::unique_ptr<Population> m_population;

    std::unique_ptr<Network> m_currentNet;
    bool m_attemptActive = false;
    int m_cursor = 0;

    std::vector<int> m_replayToggles;
    size_t m_replayCursor = 0;
    bool m_replayHold = false;
    int m_takeoverStep = 0;
    int m_step = 0;
    std::vector<int> m_recordToggles;
    bool m_recordHold = false;
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

    bool m_promptPending = false;
    std::string m_promptText;

    float m_savedFpsTarget = 0.f;
    bool m_fpsOverridden = false;
};

}
