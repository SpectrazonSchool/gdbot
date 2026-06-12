#pragma once

// Neat - minimal, self-contained NEAT (NeuroEvolution of Augmenting
// Topologies) implementation. Pure C++, no Geode/Cocos includes, so it is
// safe to use from the background training thread.
//
// Genomes are constrained to feed-forward topologies: every node carries a
// "depth" in [0, 1] and connections may only go from lower to higher depth,
// so network evaluation is a single pass in depth order (no cycle handling).

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

namespace neatgd {

struct ConnGene {
    int in = 0;
    int out = 0;
    double weight = 0.0;
    bool enabled = true;
    int innovation = 0;
};

// Node id layout: [0, numInputs) inputs, numInputs = bias (constant 1),
// then numOutputs outputs, then hidden nodes in creation order.
struct Genome {
    int numInputs = 0;
    int numOutputs = 0;
    int nodeCount = 0;            // == max node id + 1
    std::vector<double> depths;   // per node id, inputs 0.0, outputs 1.0
    std::vector<ConnGene> conns;  // kept sorted by innovation
    double fitness = 0.0;

    // Action tape of this genome's last attempt: the decision steps (outer
    // PlayerObject::update calls since attempt start, one per engine
    // substep) at which the held button state flipped, starting from
    // released, plus the step the attempt ended on. Children replay their
    // parent's tape up to just before the parent's frontier, so mutation
    // only expresses behaviour near where the parent died - sections
    // already passed stay passed. The tape is open-loop (pure button
    // history through a deterministic engine), so it remains valid no
    // matter how the network genes are mutated or crossed over.
    // Replay/recording lives in NEATManager.
    std::vector<int> tapeToggles;
    int reachStep = 0;

    int outputId(int i) const { return numInputs + 1 + i; }
};

// Flattened evaluator for one genome. Build once, eval many times.
class Network {
public:
    explicit Network(Genome const& genome);

    // inputs.size() must be numInputs. Returns output 0 in (0, 1).
    double eval(std::vector<double> const& inputs);

private:
    struct Link {
        int from;
        double weight;
    };
    struct Node {
        int id;
        std::vector<Link> incoming;
    };
    int m_numInputs;
    std::vector<Node> m_order;     // non-input nodes in depth order
    std::vector<double> m_values;  // by node id
    int m_outputId;
};

class Population {
public:
    Population(int size, int numInputs, int numOutputs, uint32_t seed);

    int size() const { return static_cast<int>(m_genomes.size()); }
    Genome const& genome(int i) const { return m_genomes[i]; }
    void setFitness(int i, double fitness) { m_genomes[i].fitness = fitness; }
    // Persist a genome's recorded attempt history (see Genome::tapeToggles).
    void setTape(int i, std::vector<int> toggles, int reachStep) {
        m_genomes[i].tapeToggles = std::move(toggles);
        m_genomes[i].reachStep = reachStep;
    }

    // Best genome of the current (already-scored) generation.
    Genome const& best() const;

    // Speciate + reproduce: replaces the population with the next generation.
    // With `explore` set (use when fitness has stagnated), mutation rates are
    // raised and a slice of the population is replaced by fresh random
    // genomes, to break out of converged local optima. If `elite` is given it
    // is copied verbatim into the next generation, guaranteeing the best
    // genome survives unchanged across the boundary.
    void epoch(bool explore = false, Genome const* elite = nullptr);

private:
    int innovationFor(int in, int out);
    // Randomize a genome's weights, seeding the bias->output connection
    // strongly negative so the network defaults to "no jump".
    void randomizeWeights(Genome& g);
    Genome freshGenome();
    void mutate(Genome& g, bool explore);
    void mutateWeights(Genome& g, bool explore);
    bool mutateAddConnection(Genome& g);
    void mutateAddNode(Genome& g);
    Genome crossover(Genome const& fitter, Genome const& other);
    static double compatibility(Genome const& a, Genome const& b);
    // Bound a genome's accumulated *disabled* connections (addNode only ever
    // grows a genome and disabled genes are otherwise never removed), so the
    // O(conns) operations don't creep up over a long run.
    static void pruneDisabled(Genome& g);

    double rand01() { return m_uniform(m_rng); }
    double randWeight() { return m_uniform(m_rng) * 4.0 - 2.0; }

    std::vector<Genome> m_genomes;
    // Minimal fully-connected genome (weights zeroed), template for fresh
    // random genomes injected during exploration.
    Genome m_proto;
    std::mt19937 m_rng;
    std::uniform_real_distribution<double> m_uniform{0.0, 1.0};
    std::normal_distribution<double> m_gauss{0.0, 0.5};
    // (in, out) -> innovation number, persistent so identical structural
    // mutations get identical innovations across the whole run. Keyed by the
    // two node ids packed into one 64-bit value for O(1) lookup - a linear
    // scan here grew with the run and was a measurable per-epoch drag.
    std::unordered_map<int64_t, int> m_innovations;
    int m_nextInnovation = 0;
};

} // namespace neatgd
