#include "Neat.hpp"

#include <Geode/Geode.hpp>

#include <algorithm>
#include <cmath>
#include <map>

using namespace geode::prelude;

namespace neatgd {

namespace {

// Steepened sigmoid, the standard NEAT activation.
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-4.9 * x));
}

// NEAT compatibility coefficients (Stanley & Miikkulainen defaults).
constexpr double C_EXCESS = 1.0;
constexpr double C_DISJOINT = 1.0;
constexpr double C_WEIGHT = 0.4;
constexpr double COMPAT_THRESHOLD = 3.0;

constexpr double PROB_MUTATE_WEIGHTS = 0.8;
constexpr double PROB_PERTURB = 0.9; // vs. full replacement, per gene
constexpr double PROB_ADD_CONN = 0.08;
constexpr double PROB_ADD_NODE = 0.03;
constexpr double PROB_MUTATE_ONLY = 0.25;
constexpr double PROB_INHERIT_DISABLED = 0.75;
constexpr double WEIGHT_CLAMP = 8.0;

// Most disabled connections a genome may carry. addNode disables one
// connection and adds two, and nothing else removes disabled genes, so over a
// long run conns grows without bound - inflating every O(conns) operation
// (compatibility, crossover, the Network build, plain copies). When a genome
// exceeds the budget the oldest disabled genes are dropped first: recently
// disabled ones sit nearer the active frontier and are the ones crossover can
// usefully re-enable, so they're worth keeping. Disabled genes are skipped by
// the evaluator anyway, so dropping them never changes a network's behaviour.
constexpr size_t MAX_DISABLED_CONNS = 30;

} // namespace

// --- Network ---------------------------------------------------------------

Network::Network(Genome const& genome)
    : m_numInputs(genome.numInputs),
      m_outputId(genome.outputId(0)) {
    m_values.assign(genome.nodeCount, 0.0);

    std::vector<int> order;
    for (int id = genome.numInputs + 1; id < genome.nodeCount; ++id) {
        order.push_back(id);
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return genome.depths[a] < genome.depths[b];
    });

    m_order.reserve(order.size());
    for (int id : order) m_order.push_back({id, {}});

    for (auto const& c : genome.conns) {
        if (!c.enabled) continue;
        for (auto& node : m_order) {
            if (node.id == c.out) {
                node.incoming.push_back({c.in, c.weight});
                break;
            }
        }
    }
}

double Network::eval(std::vector<double> const& inputs) {
    for (int i = 0; i < m_numInputs; ++i) m_values[i] = inputs[i];
    m_values[m_numInputs] = 1.0; // bias

    for (auto const& node : m_order) {
        double sum = 0.0;
        for (auto const& link : node.incoming) {
            sum += m_values[link.from] * link.weight;
        }
        m_values[node.id] = sigmoid(sum);
    }
    return m_values[m_outputId];
}

// --- Population ------------------------------------------------------------

Population::Population(int size, int numInputs, int numOutputs, uint32_t seed)
    : m_rng(seed) {
    m_proto.numInputs = numInputs;
    m_proto.numOutputs = numOutputs;
    m_proto.nodeCount = numInputs + 1 + numOutputs;
    m_proto.depths.assign(m_proto.nodeCount, 0.0);
    for (int o = 0; o < numOutputs; ++o) {
        m_proto.depths[m_proto.outputId(o)] = 1.0;
    }
    // Fully connect inputs + bias to every output.
    for (int in = 0; in <= numInputs; ++in) {
        for (int o = 0; o < numOutputs; ++o) {
            ConnGene gene;
            gene.in = in;
            gene.out = m_proto.outputId(o);
            gene.innovation = innovationFor(gene.in, gene.out);
            m_proto.conns.push_back(gene);
        }
    }

    m_genomes.assign(size, m_proto);
    for (auto& g : m_genomes) randomizeWeights(g);
}

// Node id layout puts the bias at id == numInputs, so a connection whose `in`
// is that id is the bias->output link. Seeding it strongly negative makes a
// fresh network's resting output (scan inputs ~0 on a clear path) sit below
// the press threshold, so the default behaviour is to do nothing and only the
// *jumps* have to be learned. Without it roughly half of all fresh genomes
// hold the button from the first frame and never learn to wait.
void Population::randomizeWeights(Genome& g) {
    for (auto& c : g.conns) {
        c.weight = c.in == g.numInputs ? -2.0 + m_gauss(m_rng) : randWeight();
    }
}

Genome Population::freshGenome() {
    Genome g = m_proto;
    randomizeWeights(g);
    return g;
}

int Population::innovationFor(int in, int out) {
    int64_t const key =
        (static_cast<int64_t>(in) << 32) | static_cast<uint32_t>(out);
    auto const it = m_innovations.find(key);
    if (it != m_innovations.end()) return it->second;
    int const innov = m_nextInnovation++;
    m_innovations.emplace(key, innov);
    return innov;
}

Genome const& Population::best() const {
    int bestIdx = 0;
    for (int i = 1; i < size(); ++i) {
        if (m_genomes[i].fitness > m_genomes[bestIdx].fitness) bestIdx = i;
    }
    return m_genomes[bestIdx];
}

double Population::compatibility(Genome const& a, Genome const& b) {
    size_t ia = 0, ib = 0;
    int matching = 0, disjoint = 0, excess = 0;
    double weightDiff = 0.0;
    while (ia < a.conns.size() && ib < b.conns.size()) {
        int innovA = a.conns[ia].innovation;
        int innovB = b.conns[ib].innovation;
        if (innovA == innovB) {
            weightDiff += std::abs(a.conns[ia].weight - b.conns[ib].weight);
            ++matching;
            ++ia;
            ++ib;
        } else if (innovA < innovB) {
            ++disjoint;
            ++ia;
        } else {
            ++disjoint;
            ++ib;
        }
    }
    excess = static_cast<int>((a.conns.size() - ia) + (b.conns.size() - ib));

    double n = static_cast<double>(std::max(a.conns.size(), b.conns.size()));
    if (n < 20.0) n = 1.0; // small-genome normalization, per the NEAT paper
    double avgWeightDiff = matching > 0 ? weightDiff / matching : 0.0;
    return C_EXCESS * excess / n + C_DISJOINT * disjoint / n + C_WEIGHT * avgWeightDiff;
}

void Population::mutateWeights(Genome& g, bool explore) {
    // Exploring: perturb harder and fully re-roll weights more often.
    double const perturbProb = explore ? 0.75 : PROB_PERTURB;
    double const sigma = explore ? 2.5 : 1.0;
    for (auto& c : g.conns) {
        if (rand01() < perturbProb) {
            c.weight += m_gauss(m_rng) * sigma;
        } else {
            c.weight = randWeight();
        }
        c.weight = std::clamp(c.weight, -WEIGHT_CLAMP, WEIGHT_CLAMP);
    }
}

bool Population::mutateAddConnection(Genome& g) {
    for (int attempt = 0; attempt < 20; ++attempt) {
        int from = static_cast<int>(rand01() * g.nodeCount) % g.nodeCount;
        int to = static_cast<int>(rand01() * g.nodeCount) % g.nodeCount;
        // Feed-forward only: strictly increasing depth, and never into an
        // input/bias node (depth 0 == 0 is excluded by the strict <).
        if (g.depths[from] >= g.depths[to]) continue;
        bool exists = false;
        for (auto const& c : g.conns) {
            if (c.in == from && c.out == to) {
                exists = true;
                break;
            }
        }
        if (exists) continue;

        ConnGene gene;
        gene.in = from;
        gene.out = to;
        gene.weight = randWeight();
        gene.innovation = innovationFor(from, to);
        auto pos = std::lower_bound(
            g.conns.begin(), g.conns.end(), gene.innovation,
            [](ConnGene const& c, int innov) { return c.innovation < innov; });
        g.conns.insert(pos, gene);
        return true;
    }
    return false;
}

void Population::mutateAddNode(Genome& g) {
    std::vector<size_t> enabled;
    for (size_t i = 0; i < g.conns.size(); ++i) {
        if (g.conns[i].enabled) enabled.push_back(i);
    }
    if (enabled.empty()) return;
    size_t pick = enabled[static_cast<size_t>(rand01() * enabled.size()) % enabled.size()];

    int newNode = g.nodeCount++;
    g.depths.push_back((g.depths[g.conns[pick].in] + g.depths[g.conns[pick].out]) / 2.0);
    g.conns[pick].enabled = false;
    int in = g.conns[pick].in;
    int out = g.conns[pick].out;
    double weight = g.conns[pick].weight;

    ConnGene a;
    a.in = in;
    a.out = newNode;
    a.weight = 1.0;
    a.innovation = innovationFor(in, newNode);
    ConnGene b;
    b.in = newNode;
    b.out = out;
    b.weight = weight;
    b.innovation = innovationFor(newNode, out);
    for (auto gene : {a, b}) {
        auto pos = std::lower_bound(
            g.conns.begin(), g.conns.end(), gene.innovation,
            [](ConnGene const& c, int innov) { return c.innovation < innov; });
        g.conns.insert(pos, gene);
    }
}

void Population::pruneDisabled(Genome& g) {
    size_t disabled = 0;
    for (auto const& c : g.conns) {
        if (!c.enabled) ++disabled;
    }
    if (disabled <= MAX_DISABLED_CONNS) return;

    // conns is kept sorted by innovation ascending, so removing the
    // first-encountered disabled genes drops the oldest ones.
    size_t toRemove = disabled - MAX_DISABLED_CONNS;
    g.conns.erase(
        std::remove_if(
            g.conns.begin(), g.conns.end(),
            [&](ConnGene const& c) {
                if (!c.enabled && toRemove > 0) {
                    --toRemove;
                    return true;
                }
                return false;
            }),
        g.conns.end());
}

void Population::mutate(Genome& g, bool explore) {
    double const structBoost = explore ? 3.0 : 1.0;
    if (rand01() < PROB_MUTATE_WEIGHTS) mutateWeights(g, explore);
    if (rand01() < PROB_ADD_CONN * structBoost) mutateAddConnection(g);
    if (rand01() < PROB_ADD_NODE * structBoost) mutateAddNode(g);
    pruneDisabled(g);
}

Genome Population::crossover(Genome const& fitter, Genome const& other) {
    // Topology (nodes, depths, disjoint/excess genes) comes from the fitter
    // parent; matching genes pick their weight from either parent at random.
    Genome child = fitter;
    size_t io = 0;
    for (auto& c : child.conns) {
        while (io < other.conns.size() && other.conns[io].innovation < c.innovation) ++io;
        if (io < other.conns.size() && other.conns[io].innovation == c.innovation) {
            if (rand01() < 0.5) c.weight = other.conns[io].weight;
            if (!c.enabled || !other.conns[io].enabled) {
                c.enabled = rand01() >= PROB_INHERIT_DISABLED;
            }
        }
    }
    child.fitness = 0.0;
    return child;
}

void Population::epoch(bool explore, Genome const* elite) {
    // --- Speciate: first member of each species is its representative.
    std::vector<std::vector<int>> species;
    for (int i = 0; i < size(); ++i) {
        bool placed = false;
        for (auto& s : species) {
            if (compatibility(m_genomes[i], m_genomes[s.front()]) < COMPAT_THRESHOLD) {
                s.push_back(i);
                placed = true;
                break;
            }
        }
        if (!placed) species.push_back({i});
    }

    // Sort each species by fitness, best first.
    for (auto& s : species) {
        std::sort(s.begin(), s.end(), [&](int a, int b) {
            return m_genomes[a].fitness > m_genomes[b].fitness;
        });
    }

    // --- Offspring allotment by species' share of total adjusted fitness.
    double totalAdj = 0.0;
    std::vector<double> speciesAdj(species.size(), 0.0);
    for (size_t si = 0; si < species.size(); ++si) {
        for (int gi : species[si]) {
            speciesAdj[si] += m_genomes[gi].fitness / species[si].size();
        }
        totalAdj += speciesAdj[si];
    }

    int popSize = size();
    std::vector<int> allotted(species.size(), 0);
    int assigned = 0;
    for (size_t si = 0; si < species.size(); ++si) {
        allotted[si] = totalAdj > 0.0
            ? static_cast<int>(speciesAdj[si] / totalAdj * popSize)
            : popSize / static_cast<int>(species.size());
        assigned += allotted[si];
    }
    // Hand rounding leftovers to the best species.
    size_t bestSpecies = 0;
    for (size_t si = 1; si < species.size(); ++si) {
        if (speciesAdj[si] > speciesAdj[bestSpecies]) bestSpecies = si;
    }
    allotted[bestSpecies] += popSize - assigned;

    // --- Reproduce.
    std::vector<Genome> next;
    next.reserve(popSize);
    for (size_t si = 0; si < species.size(); ++si) {
        auto const& members = species[si];
        int quota = allotted[si];
        if (quota <= 0) continue;

        // Champion of a non-trivial species survives unchanged.
        if (members.size() >= 5) {
            next.push_back(m_genomes[members.front()]);
            --quota;
        }

        // Parents come from the top half.
        size_t parentPool = std::max<size_t>(1, members.size() / 2);
        auto pickParent = [&]() -> Genome const& {
            size_t idx = static_cast<size_t>(rand01() * parentPool) % parentPool;
            return m_genomes[members[idx]];
        };

        for (int k = 0; k < quota; ++k) {
            Genome child;
            if (members.size() == 1 || rand01() < PROB_MUTATE_ONLY) {
                child = pickParent();
                child.fitness = 0.0;
            } else {
                Genome const& a = pickParent();
                Genome const& b = pickParent();
                child = a.fitness >= b.fitness ? crossover(a, b) : crossover(b, a);
            }
            mutate(child, explore);
            next.push_back(std::move(child));
        }
    }
    // Rounding edge cases: top up / trim to exact population size.
    while (static_cast<int>(next.size()) < popSize) {
        Genome child = best();
        mutate(child, explore);
        next.push_back(std::move(child));
    }
    next.resize(popSize);

    // Exploring: replace the tail fifth with brand-new random genomes.
    if (explore) {
        int const fresh = std::max(1, popSize / 5);
        for (int i = popSize - fresh; i < popSize; ++i) {
            next[i] = freshGenome();
        }
    }

    // Hall-of-fame elitism: drop the all-time best in unchanged (overwriting a
    // fresh/random slot rather than an evolved one). fitness is reset so it
    // competes on its merits next generation.
    if (elite && elite->nodeCount > 0) {
        next.back() = *elite;
        next.back().fitness = 0.0;
    }

    m_genomes = std::move(next);
}

} // namespace neatgd
