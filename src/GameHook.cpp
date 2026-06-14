#include "GameHook.hpp"
#include "NEATManager.hpp"

#include <Geode/Geode.hpp>
#include <Geode/modify/CCAnimate.hpp>
#include <Geode/modify/CCDirector.hpp>
#include <Geode/modify/CCParticleSystem.hpp>
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

constexpr float FLOOR_TOP = 90.f;
constexpr float PLAYER_HALF = 15.f;

bool isSolidType(GameObjectType type) {
    return type == GameObjectType::Solid || type == GameObjectType::Slope;
}

bool isHazardType(GameObjectType type) {
    return type == GameObjectType::Hazard
        || type == GameObjectType::AnimatedHazard;
}

}

std::vector<LevelObject> extractLevelObjects(PlayLayer* layer) {
    std::vector<LevelObject> objects;
    if (!layer || !layer->m_objects) return objects;

    for (auto obj : CCArrayExt<GameObject*>(layer->m_objects)) {
        if (!obj) continue;
        if (obj == layer->m_player1 || obj == layer->m_player2) continue;

        auto const type = obj->m_objectType;
        bool const solid = isSolidType(type);
        bool const hazard = isHazardType(type);
        if (!solid && !hazard) continue;

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

static bool trainingGraphicsHidden() {
    auto mgr = NEATManager::get();
    return mgr->phase() == NEATManager::Phase::Training && mgr->hideGraphics();
}

}

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
        std::vector<LevelObject> cache;
        size_t scanStart = 0;
        bool holding = false;
        int jumps = 0;
        int playerCallDepth = 0;
        bool resetRequested = false;
        CCLabelBMFont* statusLabel = nullptr;
        CCLabelBMFont* centerLabel = nullptr;
        bool graphicsHidden = false;
        int frameCounter = 0;
    };

public:
    void setHolding(bool hold) {
        if (m_fields->holding == hold) return;
        m_fields->holding = hold;
        if (!m_player1) return;
        if (hold) {
            ++m_fields->jumps;
            m_player1->pushButton(PlayerButton::Jump);
        } else {
            m_player1->releaseButton(PlayerButton::Jump);
        }
    }

    size_t& playerScan() { return m_fields->scanStart; }
    bool holdingState() { return m_fields->holding; }
    int jumpCount() { return m_fields->jumps; }
    CCLabelBMFont* centerLabel() { return m_fields->centerLabel; }

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
            ceilBot[s] = headY + NORM_CLEAR;
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
                        floorTop[s] = std::max(floorTop[s], top);
                    } else if (bot >= py - 60.f) {
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
        inputs.push_back(player->m_isUpsideDown ? 1.0 : 0.0);
        inputs.push_back(player->m_isShip ? 1.0 : 0.0);
        inputs.push_back(player->m_isBird ? 1.0 : 0.0);
        inputs.push_back(player->m_isBall ? 1.0 : 0.0);
        inputs.push_back(player->m_isDart ? 1.0 : 0.0);
        inputs.push_back(player->m_isRobot ? 1.0 : 0.0);
        inputs.push_back(player->m_isSpider ? 1.0 : 0.0);
        inputs.push_back(player->m_isSwing ? 1.0 : 0.0);
        for (size_t s = 0; s < SCAN_SAMPLES; ++s) {
            inputs.push_back(std::clamp(
                (floorTop[s] - feetY) / NORM_DY, -1.f, 1.f));
        }
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

        bool const wantHidden =
            mgr->phase() == NEATManager::Phase::Training && mgr->hideGraphics();
        if (wantHidden != m_fields->graphicsHidden) {
            m_fields->graphicsHidden = wantHidden;
            if (m_objectLayer) m_objectLayer->setVisible(!wantHidden);
            if (m_background) m_background->setVisible(!wantHidden);
            if (wantHidden && !m_fields->centerLabel) {
                auto const winSize = CCDirector::sharedDirector()->getWinSize();
                m_fields->centerLabel = CCLabelBMFont::create("", "goldFont.fnt");
                if (m_fields->centerLabel) {
                    m_fields->centerLabel->setAlignment(kCCTextAlignmentCenter);
                    m_fields->centerLabel->setPosition(
                        winSize.width / 2, winSize.height / 2);
                    m_fields->centerLabel->setScale(0.9f);
                    this->addChild(m_fields->centerLabel, 9999);
                }
            }
            if (m_fields->centerLabel) {
                m_fields->centerLabel->setVisible(wantHidden);
            }
            if (wantHidden) {
                if (auto fmod = FMODAudioEngine::sharedEngine()) {
                    fmod->stopAllMusic(true);
                }
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
            this->addChild(label, 9999);
        }
        label->setVisible(true);
        if (++m_fields->frameCounter % 10 == 0) {
            label->setString(mgr->statusText().c_str());
            if (m_fields->graphicsHidden && m_fields->centerLabel) {
                m_fields->centerLabel->setString(mgr->progressText().c_str());
            }
        }
    }

    void setupHasCompleted() {
        PlayLayer::setupHasCompleted();
        m_fields->cache = extractLevelObjects(this);
        m_fields->scanStart = 0;
    }

    void resetLevel() {
        auto mgr = NEATManager::get();

        if (mgr->phase() == NEATManager::Phase::Training
            && mgr->attemptInProgress()) {
            mgr->endAttempt(this->getCurrentPercent(), true, m_fields->jumps);
        }

        PlayLayer::resetLevel();

        m_fields->scanStart = 0;
        m_fields->holding = false;
        m_fields->jumps = 0;
        m_fields->resetRequested = false;

        if (mgr->phase() == NEATManager::Phase::Training) {
            mgr->beginAttempt();
        } else if (mgr->phase() == NEATManager::Phase::Showcase) {
            mgr->onShowcaseStart();
        }
    }

    void postUpdate(float dt) {
        PlayLayer::postUpdate(dt);

        updateStatusLabel();
        auto mgr = NEATManager::get();

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

        if (mgr->phase() == NEATManager::Phase::Showcase
            && !mgr->showcaseActive() && m_player1 && !m_player1->m_isDead) {
            mgr->onShowcaseStart();
        }
    }

    void destroyPlayer(PlayerObject* player, GameObject* object) {
        bool const anticheatCall = object == m_anticheatSpike;
        auto mgr = NEATManager::get();

        if (!anticheatCall && mgr->phase() == NEATManager::Phase::Training
            && player == m_player1) {
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

        if (mgr->phase() == NEATManager::Phase::Training) {
            if (mgr->attemptInProgress()) {
                setHolding(false);
                mgr->endAttempt(100.0, false, m_fields->jumps);
            }
            m_fields->resetRequested = true;
            return;
        }

        if (mgr->showcaseActive()) {
            setHolding(false);
            mgr->onShowcaseEnd(100.0);
        }
        PlayLayer::levelComplete();
    }

    void onQuit() {
        auto mgr = NEATManager::get();
        if (mgr->isActive()) {
            setHolding(false);
            mgr->stop("level quit");
        }
        PlayLayer::onQuit();
    }
};

class $modify(NEATDirector, CCDirector) {
    void drawScene() {
        if (!trainingGraphicsHidden()) {
            CCDirector::drawScene();
            return;
        }

        this->calculateDeltaTime();
        if (!m_bPaused) {
            this->getScheduler()->update(this->getDeltaTime());
        }
        if (m_pNextScene) {
            this->setNextScene();
        }

        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        kmGLPushMatrix();
        if (auto layer = static_cast<NEATPlayLayer*>(PlayLayer::get())) {
            if (auto label = layer->centerLabel()) {
                label->visit();
            }
        }
        kmGLPopMatrix();

        ++m_uTotalFrames;

        if (m_pobOpenGLView) {
            m_pobOpenGLView->swapBuffers();
        }
    }
};

class $modify(NEATParticleSystem, CCParticleSystem) {
    void update(float dt) {
        if (trainingGraphicsHidden()) return;
        CCParticleSystem::update(dt);
    }
};

class $modify(NEATAnimate, CCAnimate) {
    void update(float t) {
        if (trainingGraphicsHidden()) return;
        CCAnimate::update(t);
    }
};

class $modify(NEATPlayerObject, PlayerObject) {
    void update(float dt) {
        auto layer = static_cast<NEATPlayLayer*>(PlayLayer::get());
        if (!layer || this != layer->m_player1) {
            PlayerObject::update(dt);
            return;
        }

        auto mgr = NEATManager::get();
        if (layer->playerCallDepth() == 0 && !this->m_isDead) {
            if (mgr->phase() == NEATManager::Phase::Training
                && mgr->attemptInProgress()) {
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
