#include "GameHook.hpp"
#include "NEATManager.hpp"
#include "PlaybackStore.hpp"

#include <Geode/Geode.hpp>
#include <Geode/modify/PauseLayer.hpp>
#include <Geode/ui/Popup.hpp>
#include <Geode/ui/ScrollLayer.hpp>
#include <Geode/ui/TextInput.hpp>

#include <algorithm>
#include <functional>
#include <utility>

using namespace geode::prelude;
using namespace neatgd;

namespace {

// Sprite-frame icon scaled to a target size, with a text-button fallback if
// the resource is missing - same pattern as the pause-menu NEAT button.
CCNode* makeIconNode(char const* frame, char const* fallback, float target) {
    CCNode* icon = CCSprite::createWithSpriteFrameName(frame);
    if (!icon) icon = ButtonSprite::create(fallback);
    if (!icon) return nullptr;
    auto const size = icon->getContentSize();
    float const largest = std::max(size.width, size.height);
    if (largest > 0.f) icon->setScale(target / largest);
    return icon;
}

} // namespace

// RenamePlaybackPopup - small text-input dialog for renaming a saved
// playback. Calls back with the new name on Save; empty names are ignored.
class RenamePlaybackPopup : public geode::Popup {
protected:
    TextInput* m_input = nullptr;
    std::function<void(std::string const&)> m_onSave;

    bool init(
        std::string const& current,
        std::function<void(std::string const&)> onSave) {
        if (!Popup::init(280.f, 140.f)) return false;
        m_onSave = std::move(onSave);
        this->setTitle("Rename Playback");

        auto const size = m_mainLayer->getContentSize();
        m_input = TextInput::create(220.f, "Playback name");
        if (!m_input) return false;
        m_input->setCommonFilter(CommonFilter::Any);
        m_input->setMaxCharCount(48);
        m_input->setString(current);
        m_input->setPosition(size.width / 2, size.height / 2 + 8.f);
        m_mainLayer->addChild(m_input);

        if (auto sprite = ButtonSprite::create("Save")) {
            auto btn = CCMenuItemSpriteExtra::create(
                sprite, this, menu_selector(RenamePlaybackPopup::onSave));
            if (btn) {
                btn->setPosition(size.width / 2, 32.f);
                m_buttonMenu->addChild(btn);
            }
        }
        return true;
    }

    void onSave(CCObject*) {
        std::string const name = m_input ? std::string(m_input->getString()) : "";
        auto const onSaveCb = m_onSave;
        this->onClose(nullptr);
        // `this` is gone past here.
        if (!name.empty() && onSaveCb) onSaveCb(name);
    }

public:
    static RenamePlaybackPopup* create(
        std::string const& current,
        std::function<void(std::string const&)> onSave) {
        auto ret = new RenamePlaybackPopup();
        if (ret->init(current, std::move(onSave))) {
            ret->autorelease();
            return ret;
        }
        delete ret;
        return nullptr;
    }
};

// PlaybackListPopup - every saved playback for the current level (one row
// per entry: name + play / rename / delete), backed by the level's .dat file
// in the mod save dir (see PlaybackStore). Opened from the settings popup;
// playing a row tears down the whole pause UI and restarts the level with
// the tape armed.
class PlaybackListPopup : public geode::Popup {
protected:
    static constexpr float LIST_W = 300.f;
    static constexpr float LIST_H = 165.f;
    static constexpr float ROW_H = 34.f;

    Ref<PauseLayer> m_pauseLayer;
    std::function<void()> m_closeSettings;
    std::string m_levelKey;
    std::vector<neatgd::Playback> m_playbacks;
    ScrollLayer* m_scroll = nullptr;
    CCLabelBMFont* m_emptyLabel = nullptr;

    bool init(
        PauseLayer* pause, std::string levelKey,
        std::function<void()> closeSettings) {
        if (!Popup::init(340.f, 240.f)) return false;
        m_pauseLayer = pause;
        m_levelKey = std::move(levelKey);
        m_closeSettings = std::move(closeSettings);
        this->setTitle("Playbacks");
        m_playbacks = neatgd::PlaybackStore::load(m_levelKey);
        this->rebuildList();
        return true;
    }

    // Rows are rebuilt from scratch after any rename/delete - the list is
    // small (one entry per finished training session) and rebuilding keeps
    // row tags trivially in sync with the vector.
    void rebuildList() {
        if (m_scroll) {
            m_scroll->removeFromParent();
            m_scroll = nullptr;
        }
        if (m_emptyLabel) {
            m_emptyLabel->removeFromParent();
            m_emptyLabel = nullptr;
        }
        auto const size = m_mainLayer->getContentSize();

        if (m_playbacks.empty()) {
            m_emptyLabel = CCLabelBMFont::create(
                "No playbacks for this level yet.\n"
                "Finish a training run first!",
                "chatFont.fnt");
            if (m_emptyLabel) {
                m_emptyLabel->setAlignment(kCCTextAlignmentCenter);
                m_emptyLabel->setPosition(size.width / 2, size.height / 2);
                m_emptyLabel->setScale(0.7f);
                m_mainLayer->addChild(m_emptyLabel);
            }
            return;
        }

        m_scroll = ScrollLayer::create(CCSize{LIST_W, LIST_H});
        if (!m_scroll) return;
        float const contentH =
            std::max(LIST_H, ROW_H * static_cast<float>(m_playbacks.size()));
        m_scroll->m_contentLayer->setContentSize({LIST_W, contentH});

        for (size_t i = 0; i < m_playbacks.size(); ++i) {
            auto const& p = m_playbacks[i];
            // Subtle alternating stripe so rows read as rows.
            auto row = CCLayerColor::create(
                ccc4(0, 0, 0, i % 2 ? 70 : 35), LIST_W, ROW_H);
            if (!row) continue;
            row->setPosition(0.f, contentH - ROW_H * (i + 1));

            auto label = CCLabelBMFont::create(p.name.c_str(), "chatFont.fnt");
            if (label) {
                label->setAnchorPoint({0.f, 0.5f});
                label->setPosition(8.f, ROW_H / 2);
                float const w = label->getContentSize().width;
                label->setScale(std::min(0.65f, w > 0.f ? 185.f / w : 0.65f));
                row->addChild(label);
            }

            auto menu = CCMenu::create();
            if (menu) {
                menu->setPosition(0.f, 0.f);
                struct Btn {
                    char const* frame;
                    char const* fallback;
                    SEL_MenuHandler handler;
                    float x;
                };
                // Order per design: play, rename, delete.
                Btn const buttons[] = {
                    {"GJ_playBtn2_001.png", ">",
                     menu_selector(PlaybackListPopup::onPlayRow), 222.f},
                    {"GJ_viewLevelsBtn_001.png", "Edit",
                     menu_selector(PlaybackListPopup::onRenameRow), 254.f},
                    {"GJ_resetBtn_001.png", "Del",
                     menu_selector(PlaybackListPopup::onDeleteRow), 286.f},
                };
                for (auto const& b : buttons) {
                    auto icon = makeIconNode(b.frame, b.fallback, 24.f);
                    if (!icon) continue;
                    auto btn = CCMenuItemSpriteExtra::create(
                        icon, this, b.handler);
                    if (!btn) continue;
                    btn->setPosition(b.x, ROW_H / 2);
                    btn->setTag(static_cast<int>(i));
                    menu->addChild(btn);
                }
                row->addChild(menu);
            }
            m_scroll->m_contentLayer->addChild(row);
        }

        m_scroll->setPosition((size.width - LIST_W) / 2, 30.f);
        m_mainLayer->addChild(m_scroll);
        m_scroll->scrollToTop();
    }

    int rowIndex(CCObject* sender) const {
        int const idx = static_cast<CCNode*>(sender)->getTag();
        return idx >= 0 && idx < static_cast<int>(m_playbacks.size()) ? idx : -1;
    }

    void onPlayRow(CCObject* sender) {
        int const idx = rowIndex(sender);
        if (idx < 0) return;
        auto playLayer = PlayLayer::get();
        if (!playLayer) return;
        if (!NEATManager::get()->beginPlayback(m_playbacks[idx])) return;

        Ref<PauseLayer> pause = m_pauseLayer;
        auto const closeSettings = m_closeSettings;
        this->onClose(nullptr);
        // `this` is gone past here.
        if (closeSettings) closeSettings();
        // Resume tears the pause layer down; the resetLevel hook then arms
        // the replay via onShowcaseStart - the exact path training uses.
        if (pause) pause->onResume(nullptr);
        playLayer->resetLevelFromStart();
    }

    void onRenameRow(CCObject* sender) {
        int const idx = rowIndex(sender);
        if (idx < 0) return;
        auto popup = RenamePlaybackPopup::create(
            m_playbacks[idx].name,
            [self = Ref(this), idx](std::string const& name) {
                if (idx >= static_cast<int>(self->m_playbacks.size())) return;
                self->m_playbacks[idx].name = name;
                neatgd::PlaybackStore::save(self->m_levelKey, self->m_playbacks);
                self->rebuildList();
            });
        if (popup) popup->show();
    }

    void onDeleteRow(CCObject* sender) {
        int const idx = rowIndex(sender);
        if (idx < 0) return;
        geode::createQuickPopup(
            "Delete Playback",
            fmt::format("Delete <cy>{}</c>?", m_playbacks[idx].name),
            "Cancel", "Delete",
            [self = Ref(this), idx](FLAlertLayer*, bool confirmed) {
                if (!confirmed) return;
                if (idx >= static_cast<int>(self->m_playbacks.size())) return;
                self->m_playbacks.erase(self->m_playbacks.begin() + idx);
                neatgd::PlaybackStore::save(self->m_levelKey, self->m_playbacks);
                self->rebuildList();
            });
    }

public:
    static PlaybackListPopup* create(
        PauseLayer* pause, std::string levelKey,
        std::function<void()> closeSettings) {
        auto ret = new PlaybackListPopup();
        if (ret->init(pause, std::move(levelKey), std::move(closeSettings))) {
            ret->autorelease();
            return ret;
        }
        delete ret;
        return nullptr;
    }
};

// TrainSettingsPopup - shown when the pause-menu button is clicked. Lets the
// user tune the training parameters, persists them as mod saved values, and
// starts live training on confirm.
class TrainSettingsPopup : public geode::Popup {
protected:
    Ref<PauseLayer> m_pauseLayer;
    TextInput* m_populationInput = nullptr;
    TextInput* m_generationsInput = nullptr;
    TextInput* m_stagnationInput = nullptr;
    TextInput* m_maxMinutesInput = nullptr;
    TextInput* m_speedInput = nullptr;
    TextInput* m_maxFpsInput = nullptr;
    CCMenuItemToggler* m_graphicsToggle = nullptr;

    TextInput* makeInput(
        char const* label, CCPoint pos, std::string const& value,
        CommonFilter filter) {
        auto input = TextInput::create(100.f, "0");
        if (!input) return nullptr;
        input->setLabel(label);
        input->setCommonFilter(filter);
        input->setMaxCharCount(7);
        input->setString(value);
        input->setPosition(pos);
        m_mainLayer->addChild(input);
        return input;
    }

    // Small info button (the standard GD circular "i") placed at the
    // top-right of a setting. Its tag selects the blurb shown by onInfo.
    void makeInfoButton(CCPoint pos, int tag) {
        CCNode* icon =
            CCSprite::createWithSpriteFrameName("GJ_infoIcon_001.png");
        if (!icon) icon = ButtonSprite::create("?"); // resource missing
        if (!icon) return;
        float const largest =
            std::max(icon->getContentSize().width, icon->getContentSize().height);
        if (largest > 0.f) icon->setScale(11.f / largest);
        auto btn = CCMenuItemSpriteExtra::create(
            icon, this, menu_selector(TrainSettingsPopup::onInfo));
        if (!btn) return;
        btn->setPosition(pos);
        btn->setTag(tag);
        m_buttonMenu->addChild(btn);
    }

    // Shown when an info button is clicked. Tags match the order the settings
    // are created in init().
    void onInfo(CCObject* sender) {
        char const* title = "";
        char const* desc = "";
        switch (sender->getTag()) {
            case 0:
                title = "Population";
                desc =
                    "How many neural networks (genomes) live in each "
                    "generation. Bigger means more variety per generation, but "
                    "fewer generations fit in the same time. NEAT improves "
                    "through generations, so a moderate <cy>150-500</c> usually "
                    "learns faster than a huge population.";
                break;
            case 1:
                title = "Generations";
                desc =
                    "The maximum number of generations to evolve before "
                    "stopping. Training can also stop earlier from the "
                    "<cy>Stagnation</c> or <cy>Minutes</c> limits. Raise this if "
                    "you want longer runs bounded by time instead.";
                break;
            case 2:
                title = "Stagnation";
                desc =
                    "Stop early if the best score hasn't improved for this many "
                    "generations in a row. It also triggers an exploration "
                    "burst (extra mutation + fresh genomes) a few generations "
                    "before giving up.";
                break;
            case 3:
                title = "Minutes";
                desc =
                    "Wall-clock time budget for the whole run. Training stops "
                    "once it's reached, whichever comes first with the other "
                    "limits. Set to <cy>0</c> to disable the time limit.";
                break;
            case 4:
                title = "Speed";
                desc =
                    "Game-speed multiplier while training, so runs finish "
                    "faster. Physics are unchanged - the engine just takes more "
                    "substeps per frame - so it doesn't affect what's learned. "
                    "Turn on <cy>Hide gfx</c> and raise <cy>Max FPS</c> to push "
                    "this higher.";
                break;
            case 6:
                title = "Max FPS";
                desc =
                    "Frame-rate target while training. The <cy>Speed</c> "
                    "multiplier only pays off if each frame's work stays under "
                    "the engine's per-frame physics limit - past that the extra "
                    "speed is silently dropped. Running more frames per second "
                    "raises that ceiling, so a high value here is what actually "
                    "delivers high <cy>Speed</c>. Best paired with <cy>Hide "
                    "gfx</c> (cheap frames). Your monitor's refresh / vsync may "
                    "cap it. Restored to your normal FPS when training ends.";
                break;
            case 5:
                title = "Hide graphics";
                desc =
                    "Blanks the level to a black screen with a progress readout "
                    "while training, so your hardware spends its budget on "
                    "physics instead of rendering. Lets you run a higher "
                    "<cy>Speed</c> value. Restored automatically when training "
                    "ends.";
                break;
            default:
                return;
        }
        if (auto alert = FLAlertLayer::create(title, desc, "OK")) {
            alert->show();
        }
    }

    bool init(PauseLayer* pause) {
        if (!Popup::init(340.f, 240.f)) return false;
        m_pauseLayer = pause;
        this->setTitle("NEAT Training");

        TrainingConfig const defaults;
        auto mod = Mod::get();
        auto const savedPop = mod->getSavedValue<int64_t>(
            "population", defaults.population);
        auto const savedGens = mod->getSavedValue<int64_t>(
            "max-generations", defaults.maxGenerations);
        auto const savedStag = mod->getSavedValue<int64_t>(
            "stagnation-limit", defaults.stagnationLimit);
        auto const savedMins = mod->getSavedValue<double>(
            "max-minutes", defaults.maxMinutes);
        auto const savedSpeed = mod->getSavedValue<double>(
            "speed", defaults.speed);
        auto const savedFps = mod->getSavedValue<double>(
            "max-fps", defaults.maxFps);

        // A row of three inputs, then a row of two.
        auto const size = m_mainLayer->getContentSize();
        float const topY = size.height / 2 + 40.f;
        float const bottomY = size.height / 2 - 20.f;

        m_populationInput = makeInput(
            "Population", {size.width / 2 - 110.f, topY},
            fmt::format("{}", savedPop), CommonFilter::Uint);
        m_generationsInput = makeInput(
            "Generations", {size.width / 2, topY},
            fmt::format("{}", savedGens), CommonFilter::Uint);
        m_stagnationInput = makeInput(
            "Stagnation", {size.width / 2 + 110.f, topY},
            fmt::format("{}", savedStag), CommonFilter::Uint);
        m_maxMinutesInput = makeInput(
            "Minutes (0=off)", {size.width / 2 - 110.f, bottomY},
            fmt::format("{:g}", savedMins), CommonFilter::Float);
        m_speedInput = makeInput(
            "Speed (x)", {size.width / 2, bottomY},
            fmt::format("{:g}", savedSpeed), CommonFilter::Float);
        m_maxFpsInput = makeInput(
            "Max FPS", {size.width / 2 + 110.f, bottomY},
            fmt::format("{:g}", savedFps), CommonFilter::Float);

        // An info button at the top-right of each field (tags match onInfo).
        float const infoDX = 46.f;
        float const infoDY = 20.f;
        makeInfoButton({size.width / 2 - 110.f + infoDX, topY + infoDY}, 0);
        makeInfoButton({size.width / 2 + infoDX, topY + infoDY}, 1);
        makeInfoButton({size.width / 2 + 110.f + infoDX, topY + infoDY}, 2);
        makeInfoButton({size.width / 2 - 110.f + infoDX, bottomY + infoDY}, 3);
        makeInfoButton({size.width / 2 + infoDX, bottomY + infoDY}, 4);
        makeInfoButton({size.width / 2 + 110.f + infoDX, bottomY + infoDY}, 6);

        // "Hide graphics" checkbox: blanks the screen while training so the
        // hardware budget goes to physics instead of rendering.
        bool const savedHide = mod->getSavedValue<bool>(
            "hide-graphics", defaults.hideGraphics);
        m_graphicsToggle = CCMenuItemToggler::createWithStandardSprites(
            this, menu_selector(TrainSettingsPopup::onToggleGraphics), 0.6f);
        if (m_graphicsToggle) {
            m_graphicsToggle->setPosition(45.f, 30.f);
            m_graphicsToggle->toggle(savedHide);
            m_buttonMenu->addChild(m_graphicsToggle);
        }
        auto hideLabel = CCLabelBMFont::create("Hide gfx", "bigFont.fnt");
        if (hideLabel) {
            hideLabel->setScale(0.35f);
            hideLabel->setAnchorPoint({0.f, 0.5f});
            hideLabel->setPosition(62.f, 30.f);
            m_mainLayer->addChild(hideLabel);
        }
        // Just left of the checkbox, clear of the "Hide gfx" label and the
        // centered Train button.
        makeInfoButton({22.f, 30.f}, 5);

        auto trainSprite = ButtonSprite::create("Train");
        if (trainSprite) {
            auto trainButton = CCMenuItemSpriteExtra::create(
                trainSprite, this, menu_selector(TrainSettingsPopup::onTrain));
            if (trainButton) {
                trainButton->setPosition(size.width / 2, 30.f);
                m_buttonMenu->addChild(trainButton);
            }
        }

        // Playback library, bottom-right of the popup - mirrors the
        // "Hide gfx" toggle's spot on the bottom-left.
        if (auto icon = makeIconNode("GJ_updateBtn_001.png", "List", 32.f)) {
            auto btn = CCMenuItemSpriteExtra::create(
                icon, this, menu_selector(TrainSettingsPopup::onPlaybacks));
            if (btn) {
                btn->setPosition(size.width - 45.f, 30.f);
                m_buttonMenu->addChild(btn);
            }
        }
        return true;
    }

    void onPlaybacks(CCObject*) {
        auto playLayer = PlayLayer::get();
        if (!playLayer || !playLayer->m_level) return;
        auto popup = PlaybackListPopup::create(
            m_pauseLayer,
            neatgd::PlaybackStore::levelKeyFor(playLayer->m_level),
            // Invoked only when a playback is played: the whole pause UI
            // (this popup included) has to come down before the restart.
            [self = Ref(this)]() { self->onClose(nullptr); });
        if (popup) popup->show();
    }

    // CCMenuItemToggler flips its own state on click; nothing to do here,
    // the value is read in onTrain.
    void onToggleGraphics(CCObject*) {}

    static int parseInt(TextInput* input, int fallback, int lo, int hi) {
        int value = fallback;
        if (input) {
            if (auto num = utils::numFromString<int>(
                    std::string(input->getString()))) {
                value = num.unwrap();
            }
        }
        return std::clamp(value, lo, hi);
    }

    static double parseDouble(TextInput* input, double fallback, double lo, double hi) {
        double value = fallback;
        if (input) {
            if (auto num = utils::numFromString<double>(
                    std::string(input->getString()))) {
                value = num.unwrap();
            }
        }
        return std::clamp(value, lo, hi);
    }

    void onTrain(CCObject*) {
        TrainingConfig const defaults;
        TrainingConfig config;
        config.population = parseInt(
            m_populationInput, defaults.population, 5, 2000);
        config.maxGenerations = parseInt(
            m_generationsInput, defaults.maxGenerations, 1, 100000);
        config.stagnationLimit = parseInt(
            m_stagnationInput, defaults.stagnationLimit, 1, 100000);
        config.maxMinutes = parseDouble(
            m_maxMinutesInput, defaults.maxMinutes, 0.0, 1440.0);
        config.speed = parseDouble(m_speedInput, defaults.speed, 1.0, 32.0);
        config.maxFps = parseDouble(m_maxFpsInput, defaults.maxFps, 60.0, 2000.0);
        config.hideGraphics =
            m_graphicsToggle && m_graphicsToggle->isToggled();

        auto mod = Mod::get();
        mod->setSavedValue<int64_t>("population", config.population);
        mod->setSavedValue<int64_t>("max-generations", config.maxGenerations);
        mod->setSavedValue<int64_t>("stagnation-limit", config.stagnationLimit);
        mod->setSavedValue<double>("max-minutes", config.maxMinutes);
        mod->setSavedValue<double>("speed", config.speed);
        mod->setSavedValue<double>("max-fps", config.maxFps);
        mod->setSavedValue<bool>("hide-graphics", config.hideGraphics);

        Ref<PauseLayer> pause = m_pauseLayer;
        this->onClose(nullptr);
        // `this` is gone past here.

        if (!pause) return;
        auto playLayer = PlayLayer::get();
        if (!playLayer) {
            log::warn("NEATGD: no PlayLayer, cannot train");
            return;
        }
        if (!NEATManager::get()->beginTraining(config)) return;

        // Resume the game (tears the pause layer down) and restart so the
        // first genome's attempt begins from level start. The resetLevel
        // hook activates it; postUpdate has a fallback if that's skipped.
        pause->onResume(nullptr);
        playLayer->resetLevelFromStart();
    }

public:
    static TrainSettingsPopup* create(PauseLayer* pause) {
        auto ret = new TrainSettingsPopup();
        if (ret->init(pause)) {
            ret->autorelease();
            return ret;
        }
        delete ret;
        return nullptr;
    }
};

// PauseLayer::init(bool) is `win inline` in bindings/2.2081 and therefore
// not hookable on Windows; customSetup (win 0x37c870) is the standard hook
// point - it is also exactly where geode.node-ids (a required dependency,
// hooked at very-early-post priority) assigns its IDs, so by the time our
// code after the original call runs, "left-button-menu" exists.
class $modify(NEATPauseLayer, PauseLayer) {
    void customSetup() {
        PauseLayer::customSetup();

        auto menuNode = this->getChildByID("left-button-menu");
        if (!menuNode) {
            log::warn("NEATGD: left-button-menu not found on PauseLayer");
            return;
        }
        auto menu = typeinfo_cast<CCMenu*>(menuNode);
        if (!menu) {
            log::warn("NEATGD: left-button-menu is not a CCMenu");
            return;
        }

        CCNode* sprite = CCSprite::create("itzar.png"_spr);
        if (!sprite) {
            log::warn("NEATGD: itzar.png missing, falling back to text button");
            sprite = ButtonSprite::create("NEAT");
        }
        if (!sprite) return;

        // Match the size of the other buttons in the left column.
        auto const size = sprite->getContentSize();
        float const largest = std::max(size.width, size.height);
        if (largest > 0.f) sprite->setScale(40.f / largest);

        auto button = CCMenuItemSpriteExtra::create(
            sprite, this, menu_selector(NEATPauseLayer::onNEAT));
        if (!button) return;
        button->setID("neat-button"_spr);

        // left-button-menu has a ColumnLayout (assigned by node-ids), so a
        // relayout slots the button into the column - no manual positioning.
        menu->addChild(button);
        menu->updateLayout();
    }

    void onNEAT(CCObject*) {
        auto mgr = NEATManager::get();
        // Clicking the button while training/showcasing cancels it.
        if (mgr->isActive()) {
            mgr->stop("cancelled from pause menu");
            return;
        }
        if (auto popup = TrainSettingsPopup::create(this)) {
            popup->show();
        }
    }
};
