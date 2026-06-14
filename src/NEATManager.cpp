#include "NEATManager.hpp"
#include "GameHook.hpp"

#include <Geode/Geode.hpp>
#include <Geode/ui/Notification.hpp>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <random>

using namespace geode::prelude;

namespace neatgd {

namespace {
constexpr double SOLVED_FITNESS = 99.0;
constexpr double PRESS_THRESHOLD = 0.55;
constexpr double RELEASE_THRESHOLD = 0.45;
constexpr double JUMP_PENALTY = 0.02;
constexpr int TAKEOVER_LEAD_MIN = 120;
constexpr int TAKEOVER_LEAD_MAX = 600;
}

NEATManager* NEATManager::get() {
    static NEATManager instance;
    return &instance;
}

bool NEATManager::beginTraining(TrainingConfig config) {
    if (m_phase != Phase::Idle) {
        log::warn("NEATGD: training already active, ignoring request");
        return false;
    }

    m_config = config;
    m_population = std::make_unique<Population>(
        config.population, static_cast<int>(INPUT_COUNT), 1,
        std::random_device{}());
    m_currentNet.reset();
    m_attemptActive = false;
    m_cursor = 0;
    m_generation = 0;
    m_sinceImproved = 0;
    m_bestFitness = 0.0;
    m_generationBest = 0.0;
    m_solved = false;
    m_bestGenome = Genome{};
    m_showcaseNet.reset();
    m_showcaseActive = false;
    m_promptPending = false;
    m_resumeSourceId = 0;
    m_trainStart = std::chrono::steady_clock::now();

    m_phase = Phase::Training;
    applyTrainingFps();

    log::info(
        "NEATGD: live sequential training started - population {}, max {} "
        "generations, stagnation limit {}, budget {:.1f}min, speed {:.1f}x",
        config.population, config.maxGenerations, config.stagnationLimit,
        config.maxMinutes, config.speed);
    return true;
}

void NEATManager::applyTrainingFps() {
    auto gm = GameManager::sharedState();
    if (!gm || m_config.maxFps <= 0.0) return;
    m_savedFpsTarget = gm->m_customFPSTarget;
    m_fpsOverridden = true;
    gm->m_customFPSTarget = static_cast<float>(m_config.maxFps);
    gm->updateCustomFPS();
    log::info(
        "NEATGD: frame-rate target raised to {:.0f} for training (was {:.0f})",
        m_config.maxFps, m_savedFpsTarget);
}

void NEATManager::restoreFps() {
    if (!m_fpsOverridden) return;
    m_fpsOverridden = false;
    if (auto gm = GameManager::sharedState()) {
        gm->m_customFPSTarget = m_savedFpsTarget;
        gm->updateCustomFPS();
        log::info("NEATGD: frame-rate target restored to {:.0f}",
                  m_savedFpsTarget);
    }
}

void NEATManager::stop(char const* reason) {
    if (m_phase == Phase::Idle) return;
    log::info("NEATGD: stopped ({}) - best fitness {:.1f}%", reason, m_bestFitness);
    if (m_phase == Phase::Training) savePlayback(false);
    restoreFps();
    m_phase = Phase::Idle;
    m_currentNet.reset();
    m_attemptActive = false;
    m_population.reset();
    m_showcaseNet.reset();
    m_showcaseActive = false;
    m_promptPending = false;
}

bool NEATManager::beginAttempt() {
    if (m_phase != Phase::Training || !m_population) return false;
    if (m_cursor >= m_population->size()) return false;
    Genome const& g = m_population->genome(m_cursor);
    m_currentNet = std::make_unique<Network>(g);

    m_replayToggles = g.tapeToggles;
    m_replayCursor = 0;
    m_replayHold = false;
    m_recordToggles.clear();
    m_recordHold = false;
    m_step = 0;
    int const lead = std::uniform_int_distribution<int>(
        TAKEOVER_LEAD_MIN, TAKEOVER_LEAD_MAX)(m_rng);
    m_takeoverStep = std::max(0, g.reachStep - lead);

    m_attemptActive = true;
    return true;
}

bool NEATManager::tapeStateAt(int step) {
    while (m_replayCursor < m_replayToggles.size()
           && m_replayToggles[m_replayCursor] <= step) {
        m_replayHold = !m_replayHold;
        ++m_replayCursor;
    }
    return m_replayHold;
}

bool NEATManager::shouldHoldCurrent(
    std::vector<double> const& inputs, bool holding) {
    if (!m_attemptActive || !m_currentNet) return false;
    int const step = m_step++;
    bool hold;
    if (step < m_takeoverStep) {
        hold = tapeStateAt(step);
    } else {
        double const out = m_currentNet->eval(inputs);
        hold = out > (holding ? RELEASE_THRESHOLD : PRESS_THRESHOLD);
    }
    if (hold != m_recordHold) {
        m_recordHold = hold;
        m_recordToggles.push_back(step);
    }
    return hold;
}

void NEATManager::endAttempt(double percent, bool died, int jumps) {
    if (m_phase != Phase::Training || !m_population) return;
    if (!m_attemptActive) return;
    m_attemptActive = false;
    m_currentNet.reset();

    m_population->setTape(m_cursor, std::move(m_recordToggles), m_step);
    m_recordToggles.clear();

    m_population->setFitness(
        m_cursor, std::max(0.0, percent - JUMP_PENALTY * jumps));
    m_generationBest = std::max(m_generationBest, percent);
    if (percent > m_bestFitness) {
        m_bestFitness = percent;
        m_bestGenome = m_population->genome(m_cursor);
        m_sinceImproved = -1;
    }
    if (!died && percent >= SOLVED_FITNESS) {
        log::info("NEATGD: a genome survived to the end of the level!");
        m_solved = true;
    }

    ++m_cursor;

    if (m_solved) {
        finishTraining(true, "level beaten!");
        return;
    }
    if (m_cursor < m_population->size()) return;

    double const elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - m_trainStart).count();
    ++m_generation;
    m_sinceImproved = m_sinceImproved < 0 ? 0 : m_sinceImproved + 1;

    log::info(
        "NEATGD: generation {}/{} - best {:.1f}%, all-time {:.1f}% "
        "({:.0f}s elapsed)",
        m_generation, m_config.maxGenerations, m_generationBest,
        m_bestFitness, elapsed);
    m_generationBest = 0.0;

    if (m_generation >= m_config.maxGenerations) {
        log::info("NEATGD: generation limit reached");
        finishTraining(false, "generation limit reached");
        return;
    }
    if (m_sinceImproved >= m_config.stagnationLimit) {
        log::warn(
            "NEATGD: no improvement for {} generations, stopping early at "
            "{:.1f}%",
            m_config.stagnationLimit, m_bestFitness);
        finishTraining(
            false,
            fmt::format(
                "stagnant for {} generations", m_config.stagnationLimit)
                .c_str());
        return;
    }
    if (m_config.maxMinutes > 0.0 && elapsed >= m_config.maxMinutes * 60.0) {
        log::warn(
            "NEATGD: time budget of {:.1f}min exhausted, stopping at {:.1f}%",
            m_config.maxMinutes, m_bestFitness);
        finishTraining(false, "time budget exhausted");
        return;
    }

    bool const explore = m_sinceImproved >= 3;
    if (explore) {
        log::info(
            "NEATGD: stagnant for {} generations - boosting mutation and "
            "injecting fresh genomes",
            m_sinceImproved);
    }
    m_population->epoch(explore, m_bestFitness > 0.0 ? &m_bestGenome : nullptr);
    m_cursor = 0;
}

void NEATManager::finishTraining(bool solved, char const* reason) {
    restoreFps();
    savePlayback(true);

    m_population.reset();
    m_currentNet.reset();
    m_attemptActive = false;

    if (m_bestFitness <= 0.0) {
        log::warn("NEATGD: training produced no usable genome");
        if (auto notif = Notification::create(
                fmt::format("NEAT done ({}) - no usable genome", reason),
                NotificationIcon::Info, 4.f)) {
            notif->show();
        }
        m_phase = Phase::Idle;
        return;
    }

    m_phase = Phase::Idle;
    m_promptText = fmt::format(
        "Training finished{} (<cy>{}</c>)\n"
        "Best run: <cg>{:.1f}%</c>\n\n"
        "Watch the playback now?\n"
        "<cj>It's also saved to this level's playback list.</c>",
        solved ? " - <cg>level beaten!</c>" : "", reason, m_bestFitness);
    m_promptPending = true;
    log::info(
        "NEATGD: training finished{} - best {:.1f}%, prompting for playback",
        solved ? " (level beaten)" : "", m_bestFitness);
}

void NEATManager::savePlayback(bool completed) {
    auto playLayer = PlayLayer::get();
    if (!playLayer || !playLayer->m_level) return;
    auto const key = PlaybackStore::levelKeyFor(playLayer->m_level);

    if (m_bestFitness <= 0.0 || m_bestGenome.reachStep <= 0) {
        if (m_resumeSourceId != 0) {
            PlaybackStore::deleteSession(key, m_resumeSourceId);
            m_resumeSourceId = 0;
        }
        return;
    }

    Playback p;
    p.timestamp = static_cast<int64_t>(std::time(nullptr));
    p.percent = m_bestFitness;
    p.reachStep = m_bestGenome.reachStep;
    p.toggles = m_bestGenome.tapeToggles;
    p.completed = completed;

    std::time_t const t = static_cast<std::time_t>(p.timestamp);
    std::tm tm{};
#ifdef GEODE_IS_WINDOWS
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32] = {};
    std::strftime(buf, sizeof buf, "%Y-%m-%d %H:%M", &tm);
    p.name = fmt::format("{} ({:.1f}%)", buf, m_bestFitness);

    auto list = PlaybackStore::load(key);
    if (m_resumeSourceId != 0) {
        list.erase(
            std::remove_if(
                list.begin(), list.end(),
                [&](Playback const& e) {
                    return e.timestamp == m_resumeSourceId;
                }),
            list.end());
        PlaybackStore::deleteSession(key, m_resumeSourceId);
        m_resumeSourceId = 0;
    }

    if (!completed && !writeSession(key, p.timestamp)) {
        log::warn("NEATGD: failed to write resume session for level {}", key);
    }

    list.push_back(std::move(p));
    if (PlaybackStore::save(key, list)) {
        log::info("NEATGD: playback saved to library for level {}", key);
    }
}

bool NEATManager::writeSession(std::string const& levelKey, int64_t id) const {
    if (!m_population) return false;
    auto const path = PlaybackStore::sessionFileFor(levelKey, id);
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;

    constexpr uint32_t SESSION_MAGIC = 0x5345474E;
    constexpr uint32_t SESSION_VERSION = 1;
    auto writePod = [&out](auto const& value) {
        out.write(reinterpret_cast<char const*>(&value), sizeof(value));
    };

    writePod(SESSION_MAGIC);
    writePod(SESSION_VERSION);

    writePod(m_config.population);
    writePod(m_config.maxGenerations);
    writePod(m_config.stagnationLimit);
    writePod(m_config.maxMinutes);
    writePod(m_config.speed);
    writePod(m_config.maxFps);
    writePod(static_cast<uint8_t>(m_config.hideGraphics ? 1 : 0));

    writePod(m_generation);
    writePod(m_sinceImproved);
    writePod(m_bestFitness);
    writePod(m_generationBest);
    writePod(m_cursor);

    writeGenome(out, m_bestGenome);
    m_population->writeState(out);

    return out.good();
}

bool NEATManager::resumeTraining(
    std::string const& levelKey, int64_t sessionId) {
    if (m_phase != Phase::Idle) return false;

    auto const path = PlaybackStore::sessionFileFor(levelKey, sessionId);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        log::warn("NEATGD: resume session missing for level {}", levelKey);
        return false;
    }

    constexpr uint32_t SESSION_MAGIC = 0x5345474E;
    constexpr uint32_t SESSION_VERSION = 1;
    auto readPod = [&in](auto& value) {
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
        return static_cast<bool>(in);
    };

    uint32_t magic = 0, version = 0;
    if (!readPod(magic) || magic != SESSION_MAGIC) return false;
    if (!readPod(version) || version != SESSION_VERSION) return false;

    TrainingConfig config;
    uint8_t hideGraphics = 0;
    if (!readPod(config.population) || !readPod(config.maxGenerations)
        || !readPod(config.stagnationLimit) || !readPod(config.maxMinutes)
        || !readPod(config.speed) || !readPod(config.maxFps)
        || !readPod(hideGraphics)) {
        return false;
    }
    config.hideGraphics = hideGraphics != 0;

    int generation = 0, sinceImproved = 0, cursor = 0;
    double bestFitness = 0.0, generationBest = 0.0;
    Genome bestGenome;
    if (!readPod(generation) || !readPod(sinceImproved)
        || !readPod(bestFitness) || !readPod(generationBest)
        || !readPod(cursor) || !readGenome(in, bestGenome)) {
        return false;
    }

    auto population = std::make_unique<Population>(
        std::max(1, config.population), static_cast<int>(INPUT_COUNT), 1,
        std::random_device{}());
    if (!population->readState(in)) {
        log::warn("NEATGD: resume session corrupt for level {}", levelKey);
        return false;
    }

    m_config = config;
    m_population = std::move(population);
    m_currentNet.reset();
    m_attemptActive = false;
    m_cursor = std::clamp(cursor, 0, std::max(0, m_population->size() - 1));
    m_generation = generation;
    m_sinceImproved = sinceImproved;
    m_bestFitness = bestFitness;
    m_generationBest = generationBest;
    m_solved = false;
    m_bestGenome = std::move(bestGenome);
    m_showcaseNet.reset();
    m_showcaseActive = false;
    m_promptPending = false;
    m_resumeSourceId = sessionId;
    m_trainStart = std::chrono::steady_clock::now();

    m_phase = Phase::Training;
    applyTrainingFps();

    log::info(
        "NEATGD: resumed training for level {} at generation {}, best {:.1f}%",
        levelKey, m_generation, m_bestFitness);
    return true;
}

bool NEATManager::takeFinishedPrompt(std::string& textOut) {
    if (!m_promptPending) return false;
    m_promptPending = false;
    textOut = std::move(m_promptText);
    m_promptText.clear();
    return true;
}

bool NEATManager::armShowcase() {
    if (m_phase != Phase::Idle || m_bestFitness <= 0.0
        || m_bestGenome.nodeCount <= 0) {
        return false;
    }
    m_showcaseActive = false;
    m_phase = Phase::Showcase;
    return true;
}

bool NEATManager::beginPlayback(Playback const& playback) {
    if (m_phase != Phase::Idle) return false;
    if (playback.reachStep <= 0) return false;
    m_bestGenome = Genome{};
    m_bestGenome.tapeToggles = playback.toggles;
    m_bestGenome.reachStep = playback.reachStep;
    m_bestFitness = playback.percent;
    m_showcaseActive = false;
    m_phase = Phase::Showcase;
    log::info("NEATGD: replaying saved playback '{}' ({:.1f}%)",
              playback.name, playback.percent);
    return true;
}

void NEATManager::onShowcaseStart() {
    if (m_phase != Phase::Showcase) return;
    m_showcaseNet = m_bestGenome.nodeCount > 0
        ? std::make_unique<Network>(m_bestGenome)
        : nullptr;
    m_replayToggles = m_bestGenome.tapeToggles;
    m_replayCursor = 0;
    m_replayHold = false;
    m_step = 0;
    m_takeoverStep = m_bestGenome.reachStep;
    m_showcaseActive = true;
    log::info("NEATGD: showcase attempt started (trained best {:.1f}%)",
              m_bestFitness);
}

bool NEATManager::shouldHold(std::vector<double> const& inputs, bool holding) {
    if (!m_showcaseActive) return false;
    int const step = m_step++;
    if (step < m_takeoverStep) return tapeStateAt(step);
    if (!m_showcaseNet) return false;
    double const out = m_showcaseNet->eval(inputs);
    return out > (holding ? RELEASE_THRESHOLD : PRESS_THRESHOLD);
}

void NEATManager::onShowcaseEnd(double percent) {
    if (!m_showcaseActive) return;
    m_showcaseActive = false;
    m_showcaseNet.reset();
    log::info("NEATGD: showcase finished at {:.1f}% - returning control",
              percent);
    m_phase = Phase::Idle;
}

std::string NEATManager::progressText() const {
    if (m_phase != Phase::Training || !m_population) return "";

    double const perGen = m_population->size();
    double const total = perGen * m_config.maxGenerations;
    double const done = m_generation * perGen + m_cursor;
    double const frac = total > 0.0 ? std::min(done / total, 1.0) : 0.0;
    double const elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - m_trainStart).count();

    std::string eta = "--:--";
    if (frac > 0.002 && frac < 1.0) {
        double remain = elapsed * (1.0 - frac) / frac;
        if (m_config.maxMinutes > 0.0) {
            remain = std::min(
                remain, std::max(0.0, m_config.maxMinutes * 60.0 - elapsed));
        }
        int const r = static_cast<int>(remain);
        eta = r >= 3600
            ? fmt::format("{}:{:02}:{:02}", r / 3600, (r / 60) % 60, r % 60)
            : fmt::format("{}:{:02}", r / 60, r % 60);
    }
    return fmt::format(
        "{:.1f}% complete\nETA {}\nbest {:.1f}%\nstale {}/{}",
        frac * 100.0, eta, m_bestFitness,
        std::max(m_sinceImproved, 0), m_config.stagnationLimit);
}

std::string NEATManager::statusText() const {
    switch (m_phase) {
        case Phase::Training: {
            int const popSize = m_population ? m_population->size() : 0;
            return fmt::format(
                "NEAT gen {}/{}  genome {}/{}  best {:.1f}%  stale {}/{}",
                m_generation + 1, m_config.maxGenerations,
                std::min(m_cursor + 1, std::max(popSize, 1)), popSize,
                m_bestFitness, std::max(m_sinceImproved, 0),
                m_config.stagnationLimit);
        }
        case Phase::Showcase:
            return fmt::format("NEAT showcase  best {:.1f}%", m_bestFitness);
        default:
            return "";
    }
}

}
