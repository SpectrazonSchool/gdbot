#include "GameHook.hpp"
#include "NEATManager.hpp"

#include <Geode/Geode.hpp>
#include <Geode/modify/CCScheduler.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include <Geode/ui/Popup.hpp>

#include <algorithm>
#include <cmath>

using namespace geode::prelude;
using namespace neatgd;

namespace neatgd {

namespace {

constexpr float FLOOR_TOP = 90.f;   // ground surface Y (block row 0 bottom)
constexpr float PLAYER_HALF = 15.f; // cube main hitbox is 30x30

// Enum values verified against bindings/include/Geode/Enums.hpp:
// Solid = 0, Hazard = 2, Slope = 25, AnimatedHazard = 47.
bool isSolidType(GameObjectType type) {
    return type == GameObjectType::Solid || type == GameObjectType::Slope;
}

bool isHazardType(GameObjectType type) {
    return type == GameObjectType::Hazard
        || type == GameObjectType::AnimatedHazard;
}

} // namespace

std::vector<LevelObject> extractLevelObjects(PlayLayer* layer) {
    std::vector<LevelObject> objects;
    // GJBaseGameLayer::m_objects: CCArray* of every GameObject in the level
    // (verified in bindings/2.2081/GeometryDash.bro).
    if (!layer || !layer->m_objects) return objects;

    for (auto obj : CCArrayExt<GameObject*>(layer->m_objects)) {
        if (!obj) continue;
        // The player is a GameObject too - not level geometry.
        if (obj == layer->m_player1 || obj == layer->m_player2) continue;

        auto const type = obj->m_objectType;
        bool const solid = isSolidType(type);
        bool const hazard = isHazardType(type);
        if (!solid && !hazard) continue;

        // GameObject::getObjectRect() (virtual, win 0x1976a0 in bindings/
        // 2.2081) is the object's real collision rect in level coordinates.
        // Only used for the network's *inputs* - collisions themselves are
        // the real game's business.
        auto const& rect = obj->getObjectRect();
        LevelObject lo;
        if (rect.size.width > 0.f && rect.size.height > 0.f) {
            lo.x = rect.origin.x + rect.size.width / 2.f;
            lo.y = rect.origin.y + rect.size.height / 2.f;
            lo.w = rect.size.width;
            lo.h = rect.size.height;
        } else {
            lo.x = obj->getPositionX();
            lo.y = obj->getPositionY();
        }
        lo.hazard = hazard;
        lo.slope = type == GameObjectType::Slope;
        lo.objectId = obj->m_objectID;
        objects.push_back(lo);
    }

    std::sort(
        objects.begin(), objects.end(),
        [](LevelObject const& a, LevelObject const& b) { return a.x < b.x; });
    log::info("NEATGD: cached {} collidable objects", objects.size());
    return objects;
}

// Dismiss the PauseLayer that PlayLayer::pauseGame opened (it sits as a
// direct child of the running scene); onResume is the same path the Resume
// button takes, so music/state restore matches a manual unpause.
static void resumePauseLayer() {
    auto scene = CCDirector::sharedDirector()->getRunningScene();
    if (!scene || !scene->getChildren()) return;
    for (auto child : CCArrayExt<CCNode*>(scene->getChildren())) {
        if (auto pause = typeinfo_cast<PauseLayer*>(child)) {
            pause->onResume(nullptr);
            return;
        }
    }
}

} // namespace neatgd

// Fast-forward during training by scaling the frame delta at the source.
// CCScheduler::setTimeScale alone doesn't stick (GD manages that field), so
// multiply dt in the update hook instead - the standard speedhack approach.
// Physics stay exact: GD quantizes whatever dt it receives into fixed-rate
// substeps (GJBaseGameLayer::getModifiedDelta, win 0x2377b0 in bindings/
// 2.2081), so a larger dt means more substeps per frame, not bigger steps.
// This is also the training-fidelity argument: the jump decision is made per
// outer PlayerObject::update call (see NEATPlayerObject below), which the
// engine issues at its own fixed cadence in *game* time - so a genome
// trained at N× sees the same physics and the same decision points as the
// 1× showcase.
class $modify(NEATScheduler, CCScheduler) {
    void update(float dt) {
        auto mgr = NEATManager::get();
        if (mgr->phase() == NEATManager::Phase::Training) {
            dt *= static_cast<float>(mgr->trainingSpeed());
        }
        CCScheduler::update(dt);
    }
};

class $modify(NEATPlayLayer, PlayLayer) {
    struct Fields {
        // Collidable geometry sorted by X, rebuilt once per level load.
        std::vector<LevelObject> cache;
        // Scan position for the terrain inputs; monotonic in X within an
        // attempt, reset on every level reset.
        size_t scanStart = 0;
        // Whether the player has the jump button held via pushButton.
        bool holding = false;
        // Button presses this attempt, for the jump-efficiency penalty.
        int jumps = 0;
        // > 0 while inside PlayerObject::update on the real player, so the
        // per-substep decision in the hook below fires only on the engine's
        // outer call, never on a nested re-entry.
        int playerCallDepth = 0;
        // Set when the attempt is over and the level should restart.
        bool resetRequested = false;
        CCLabelBMFont* statusLabel = nullptr;
        // Fullscreen black cover shown while graphics are hidden, with a
        // centered progress/ETA readout.
        CCLayerColor* blackout = nullptr;
        CCLabelBMFont* centerLabel = nullptr;
        bool graphicsHidden = false;
        int frameCounter = 0;
    };

public:
    void setHolding(bool hold) {
        if (m_fields->holding == hold) return;
        m_fields->holding = hold;
        if (!m_player1) return;
        // PlayerObject::pushButton/releaseButton (win 0x397f40 / 0x3981d0 in
        // bindings/2.2081) is the per-player input path - the same one a
        // real tap reaches through handleButton, so training, showcase and
        // human play all share identical input handling.
        if (hold) {
            ++m_fields->jumps; // the efficiency penalty counts presses
            m_player1->pushButton(PlayerButton::Jump);
        } else {
            m_player1->releaseButton(PlayerButton::Jump);
        }
    }

    // Scan cursor / button state for the player's network inputs.
    size_t& playerScan() { return m_fields->scanStart; }
    bool holdingState() { return m_fields->holding; }
    int jumpCount() { return m_fields->jumps; }

    // The network inputs for a player: self state plus the terrain scan
    // described in GameHook.hpp.
    std::vector<double> computeInputs(
        PlayerObject* player, bool holding, size_t& scan) {
        float const px = player->getPositionX();
        float const py = player->getPositionY();

        auto& cache = m_fields->cache;
        while (scan < cache.size()
               && cache[scan].x + cache[scan].w / 2.f < px - 30.f) {
            ++scan;
        }

        float const feetY = py - PLAYER_HALF;
        float const headY = py + PLAYER_HALF;

        float floorTop[SCAN_SAMPLES];
        float ceilBot[SCAN_SAMPLES];
        float hazUp[SCAN_SAMPLES];
        float hazDown[SCAN_SAMPLES];
        for (size_t s = 0; s < SCAN_SAMPLES; ++s) {
            floorTop[s] = FLOOR_TOP;
            ceilBot[s] = headY + NORM_CLEAR; // open sky
            hazUp[s] = NORM_CLEAR;
            hazDown[s] = NORM_CLEAR;
        }

        float const lookEnd = px + SCAN_OFFSETS[SCAN_SAMPLES - 1] + 30.f;
        for (size_t i = scan; i < cache.size(); ++i) {
            auto const& obj = cache[i];
            float const halfW = obj.w / 2.f;
            if (obj.x - halfW > lookEnd) break;

            for (size_t s = 0; s < SCAN_SAMPLES; ++s) {
                float const cx = px + SCAN_OFFSETS[s];
                if (std::abs(obj.x - cx) > halfW) continue;
                if (obj.hazard) {
                    if (obj.y >= py) {
                        hazUp[s] = std::min(hazUp[s], obj.y - py);
                    } else {
                        hazDown[s] = std::min(hazDown[s], py - obj.y);
                    }
                } else {
                    float const top = obj.y + obj.h / 2.f;
                    float const bot = obj.y - obj.h / 2.f;
                    if (top <= py + 60.f) {
                        // Standable: reachable with a jump from here.
                        floorTop[s] = std::max(floorTop[s], top);
                    } else if (bot >= py - 60.f) {
                        // Overhang/wall at this height: its underside is the
                        // ceiling of the passable gap (a wall with a gap at
                        // the bottom reads as a low ceiling - fly under it).
                        ceilBot[s] = std::min(ceilBot[s], bot);
                    }
                }
            }
        }

        std::vector<double> inputs;
        inputs.reserve(INPUT_COUNT);
        inputs.push_back((py - (FLOOR_TOP + PLAYER_HALF)) / NORM_Y);
        inputs.push_back(player->m_yVelocity / NORM_VEL);
        inputs.push_back(player->m_isOnGround ? 1.0 : 0.0);
        inputs.push_back(holding ? 1.0 : 0.0);
        // Gamemode awareness (members verified in bindings/2.2081): gravity
        // plus a one-hot of the vehicle, so the same genome can learn
        // per-mode control (tap to jump vs. hold to climb).
        inputs.push_back(player->m_isUpsideDown ? 1.0 : 0.0);
        inputs.push_back(player->m_isShip ? 1.0 : 0.0);
        inputs.push_back(player->m_isBird ? 1.0 : 0.0);
        inputs.push_back(player->m_isBall ? 1.0 : 0.0);
        inputs.push_back(player->m_isDart ? 1.0 : 0.0);
        inputs.push_back(player->m_isRobot ? 1.0 : 0.0);
        inputs.push_back(player->m_isSpider ? 1.0 : 0.0);
        inputs.push_back(player->m_isSwing ? 1.0 : 0.0);
        // Floor: signed surface height relative to the feet - 0 is flat
        // ground underfoot, positive a ledge to clear, negative a drop. This
        // already rests at ~0 on empty ground, so it's left centred.
        for (size_t s = 0; s < SCAN_SAMPLES; ++s) {
            inputs.push_back(std::clamp(
                (floorTop[s] - feetY) / NORM_DY, -1.f, 1.f));
        }
        // Ceiling and hazards: encoded as *proximity* (0 = clear corridor,
        // rising toward 1 as something intrudes), not as distance. So an open
        // stretch drives all of these to ~0 and the network's resting output
        // is just its (negatively-seeded) bias - i.e. it does nothing by
        // default and only has to learn to *act* at obstacles. The old
        // encoding mapped "clear" to 1, leaving ~24 inputs permanently firing
        // with random weights, which made a steady low output - the restraint
        // a "wait it out" section needs - almost impossible to represent.
        for (size_t s = 0; s < SCAN_SAMPLES; ++s) {
            inputs.push_back(std::clamp(
                1.f - (ceilBot[s] - headY) / NORM_CLEAR, 0.f, 1.f));
        }
        for (size_t s = 0; s < SCAN_SAMPLES; ++s) {
            inputs.push_back(std::clamp(
                1.f - hazUp[s] / NORM_CLEAR, 0.f, 1.f));
        }
        for (size_t s = 0; s < SCAN_SAMPLES; ++s) {
            inputs.push_back(std::clamp(
                1.f - hazDown[s] / NORM_CLEAR, 0.f, 1.f));
        }
        return inputs;
    }

    int& playerCallDepth() { return m_fields->playerCallDepth; }

    void updateStatusLabel() {
        auto mgr = NEATManager::get();

        // "Hide graphics" setting: skipping the object layer's visit() is
        // the actual rendering saving (it's the node tree with every level
        // object); the blackout layer just covers what remains. Restored
        // automatically when training ends or is cancelled.
        bool const wantHidden =
            mgr->phase() == NEATManager::Phase::Training && mgr->hideGraphics();
        if (wantHidden != m_fields->graphicsHidden) {
            m_fields->graphicsHidden = wantHidden;
            // GJBaseGameLayer::m_objectLayer / m_background (verified in
            // bindings/2.2081).
            if (m_objectLayer) m_objectLayer->setVisible(!wantHidden);
            if (m_background) m_background->setVisible(!wantHidden);
            if (wantHidden && !m_fields->blackout) {
                auto const winSize = CCDirector::sharedDirector()->getWinSize();
                m_fields->blackout = CCLayerColor::create(
                    {0, 0, 0, 255}, winSize.width, winSize.height);
                if (m_fields->blackout) {
                    this->addChild(m_fields->blackout, 9000);
                }
                m_fields->centerLabel = CCLabelBMFont::create("", "goldFont.fnt");
                if (m_fields->centerLabel) {
                    m_fields->centerLabel->setAlignment(kCCTextAlignmentCenter);
                    m_fields->centerLabel->setPosition(
                        winSize.width / 2, winSize.height / 2);
                    m_fields->centerLabel->setScale(0.9f);
                    this->addChild(m_fields->centerLabel, 9999);
                }
            }
            if (m_fields->blackout) {
                m_fields->blackout->setVisible(wantHidden);
            }
            if (m_fields->centerLabel) {
                m_fields->centerLabel->setVisible(wantHidden);
            }
        }

        auto& label = m_fields->statusLabel;
        if (!mgr->isActive()) {
            if (label) label->setVisible(false);
            return;
        }
        if (!label) {
            label = CCLabelBMFont::create("", "chatFont.fnt");
            if (!label) return;
            auto const winSize = CCDirector::sharedDirector()->getWinSize();
            label->setAnchorPoint({0.f, 1.f});
            label->setPosition(5.f, winSize.height - 5.f);
            label->setScale(0.6f);
            // PlayLayer itself doesn't scroll (m_objectLayer does), so the
            // label stays screen-fixed.
            this->addChild(label, 9999);
        }
        label->setVisible(true);
        if (++m_fields->frameCounter % 10 == 0) {
            label->setString(mgr->statusText().c_str());
            if (m_fields->graphicsHidden && m_fields->centerLabel) {
                m_fields->centerLabel->setString(mgr->progressText().c_str());
            }
        }
        // Silence the soundtrack while the screen is blanked. GD restarts
        // the song on every attempt, so this re-stops it periodically;
        // music comes back naturally on the showcase reset.
        // FMODAudioEngine::stopAllMusic(bool clear) - win 0x59da0 in
        // bindings/2.2081.
        if (m_fields->graphicsHidden && m_fields->frameCounter % 20 == 0) {
            if (auto fmod = FMODAudioEngine::sharedEngine()) {
                fmod->stopAllMusic(true);
            }
        }
    }

    // --- Hooks ---------------------------------------------------------

    void setupHasCompleted() {
        PlayLayer::setupHasCompleted();
        m_fields->cache = extractLevelObjects(this);
        m_fields->scanStart = 0;
    }

    // Attempt start: close out an unfinished attempt (a manual restart
    // mid-attempt counts as a death where the player stands), then arm the
    // next genome - or the showcase network.
    void resetLevel() {
        auto mgr = NEATManager::get();

        if (mgr->phase() == NEATManager::Phase::Training
            && mgr->attemptInProgress()) {
            // getCurrentPercent (win 0x3b3950 in bindings/2.2081) must be
            // read before the reset moves the player back to the start.
            // May change phase (last genome / stop checks)!
            mgr->endAttempt(this->getCurrentPercent(), true, m_fields->jumps);
        }

        PlayLayer::resetLevel();

        m_fields->scanStart = 0;
        m_fields->holding = false; // GD clears held input on reset itself
        m_fields->jumps = 0;
        m_fields->resetRequested = false;

        if (mgr->phase() == NEATManager::Phase::Training) {
            mgr->beginAttempt();
        } else if (mgr->phase() == NEATManager::Phase::Showcase) {
            mgr->onShowcaseStart();
        }
    }

    // PlayLayer::postUpdate(float) is win 0x3b4ba0 in bindings/2.2081.
    void postUpdate(float dt) {
        PlayLayer::postUpdate(dt);

        updateStatusLabel();
        auto mgr = NEATManager::get();

        // End-of-training prompt: training can finish unattended (left
        // overnight), so instead of auto-playing the one-shot showcase into
        // an empty room, pause the game and ask. Checked before the queued
        // reset below so the pause lands first; "Later" leaves the game on
        // the pause menu, and the run is already saved to the playback
        // library either way. PlayLayer::pauseGame(bool unfocused) - win
        // 0x3ba0e0 in bindings/2.2081.
        std::string prompt;
        if (mgr->takeFinishedPrompt(prompt)) {
            this->pauseGame(false);
            geode::createQuickPopup(
                "NEAT Training", prompt, "Later", "Watch",
                [](FLAlertLayer*, bool watch) {
                    if (!watch) return;
                    auto playLayer = PlayLayer::get();
                    if (!playLayer) return;
                    if (!NEATManager::get()->armShowcase()) return;
                    resumePauseLayer();
                    playLayer->resetLevelFromStart();
                });
            return;
        }

        if (m_fields->resetRequested) {
            m_fields->resetRequested = false;
            this->resetLevel();
            return;
        }

        // Input is injected per-substep by the PlayerObject::update hook;
        // here we only arm the network if resetLevel was skipped.
        if (mgr->phase() == NEATManager::Phase::Showcase
            && !mgr->showcaseActive() && m_player1 && !m_player1->m_isDead) {
            mgr->onShowcaseStart();
        }
    }

    // destroyPlayer double duty: filter the constant anticheat-spike calls
    // (GJBaseGameLayer::m_anticheatSpike), and during training capture the
    // genome's fitness at the exact moment the engine declares it dead.
    void destroyPlayer(PlayerObject* player, GameObject* object) {
        bool const anticheatCall = object == m_anticheatSpike;
        auto mgr = NEATManager::get();

        if (!anticheatCall && mgr->phase() == NEATManager::Phase::Training
            && player == m_player1) {
            // The engine just killed the genome under evaluation: this call
            // is the ground truth the old clone system approximated. Score
            // it here and restart, skipping the real death machinery - no
            // death effect, no attempt stats from AI grinding, and hazard
            // overlap re-calls this every substep until the reset lands.
            if (mgr->attemptInProgress()) {
                setHolding(false);
                mgr->endAttempt(
                    this->getCurrentPercent(), true, m_fields->jumps);
                m_fields->resetRequested = true;
            }
            return;
        }

        PlayLayer::destroyPlayer(player, object);

        if (anticheatCall || !player || !player->m_isDead) return;
        if (player != m_player1) return;

        if (mgr->showcaseActive()) {
            setHolding(false);
            mgr->onShowcaseEnd(this->getCurrentPercent());
        }
    }

    void levelComplete() {
        auto mgr = NEATManager::get();

        // A genome genuinely reaching the end during training must NOT count
        // as a real completion (no completion stats from AI runs): score it
        // and restart - endAttempt's solved check flips to Showcase, and the
        // showcase replay then goes through the real completion path.
        if (mgr->phase() == NEATManager::Phase::Training) {
            if (mgr->attemptInProgress()) {
                setHolding(false);
                mgr->endAttempt(100.0, false, m_fields->jumps);
            }
            m_fields->resetRequested = true;
            return; // skip the original - no fake completion stats
        }

        if (mgr->showcaseActive()) {
            setHolding(false);
            mgr->onShowcaseEnd(100.0);
        }
        PlayLayer::levelComplete();
    }

    // Leaving the level aborts everything; the Training phase ending also
    // drops the speed multiplier.
    void onQuit() {
        auto mgr = NEATManager::get();
        if (mgr->isActive()) {
            setHolding(false);
            mgr->stop("level quit");
        }
        PlayLayer::onQuit();
    }
};

// The per-substep decision point, shared by training and showcase - the one
// place the network meets the engine. The engine calls PlayerObject::update
// at its own fixed cadence in game time regardless of the speed multiplier
// (see the NEATScheduler comment), so deciding here gives training and the
// 1x showcase identical decision points. Deciding once per render frame
// instead would be too coarse: at speed N one frame spans N * 4+ substeps
// of game time, putting a hard floor on jump-timing precision that walls
// off anything needing a tight jump.
// PlayerObject::update(float) - win 0x388d80 in bindings/2.2081.
class $modify(NEATPlayerObject, PlayerObject) {
    void update(float dt) {
        auto layer = static_cast<NEATPlayLayer*>(PlayLayer::get());
        if (!layer || this != layer->m_player1) {
            PlayerObject::update(dt);
            return;
        }

        auto mgr = NEATManager::get();
        if (layer->playerCallDepth() == 0 && !this->m_isDead) {
            // While the inherited tape's prefix is deciding (frontier
            // exploration - see NEATManager), the network isn't consulted,
            // so skip the terrain scan entirely: late in training the
            // replayed prefix is most of each attempt. The scan cursor in
            // computeInputs is a monotonic skip-ahead, so it catches up in
            // one pass at takeover.
            if (mgr->phase() == NEATManager::Phase::Training
                && mgr->attemptInProgress()) {
                // The genome under evaluation drives the real player through
                // the full engine: training fitness IS showcase fitness, by
                // construction rather than by approximation.
                if (mgr->replaying()) {
                    layer->setHolding(
                        mgr->shouldHoldCurrent({}, layer->holdingState()));
                } else {
                    auto inputs = layer->computeInputs(
                        this, layer->holdingState(), layer->playerScan());
                    layer->setHolding(
                        mgr->shouldHoldCurrent(inputs, layer->holdingState()));
                }
            } else if (mgr->showcaseActive()) {
                if (mgr->replaying()) {
                    layer->setHolding(
                        mgr->shouldHold({}, layer->holdingState()));
                } else {
                    auto inputs = layer->computeInputs(
                        this, layer->holdingState(), layer->playerScan());
                    layer->setHolding(
                        mgr->shouldHold(inputs, layer->holdingState()));
                }
            }
        }
        ++layer->playerCallDepth();
        PlayerObject::update(dt);
        --layer->playerCallDepth();
    }
};
