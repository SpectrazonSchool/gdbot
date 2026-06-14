#include "Neat.hpp"

#include <Geode/Geode.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <ostream>
#include <sstream>

using namespace geode::prelude;

namespace neatgd {

namespace {

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-4.9 * x));
}

constexpr double C_EXCESS = 1.0;
constexpr double C_DISJOINT = 1.0;
constexpr double C_WEIGHT = 0.4;
constexpr double COMPAT_THRESHOLD = 3.0;

constexpr double PROB_MUTATE_WEIGHTS = 0.8;
constexpr double PROB_PERTURB = 0.9;
constexpr double PROB_ADD_CONN = 0.08;
constexpr double PROB_ADD_NODE = 0.03;
constexpr double PROB_MUTATE_ONLY = 0.25;
constexpr double PROB_INHERIT_DISABLED = 0.75;
constexpr double WEIGHT_CLAMP = 8.0;

constexpr size_t MAX_DISABLED_CONNS = 30;

}

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
    m_values[m_numInputs] = 1.0;

    for (auto const& node : m_order) {
        double sum = 0.0;
        for (auto const& link : node.incoming) {
            sum += m_values[link.from] * link.weight;
        }
        m_values[node.id] = sigmoid(sum);
    }
    return m_values[m_outputId];
}

Population::Population(int size, int numInputs, int numOutputs, uint32_t seed)
    : m_rng(seed) {
    m_proto.numInputs = numInputs;
    m_proto.numOutputs = numOutputs;
    m_proto.nodeCount = numInputs + 1 + numOutputs;
    m_proto.depths.assign(m_proto.nodeCount, 0.0);
    for (int o = 0; o < numOutputs; ++o) {
        m_proto.depths[m_proto.outputId(o)] = 1.0;
    }
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
    if (n < 20.0) n = 1.0;
    double avgWeightDiff = matching > 0 ? weightDiff / matching : 0.0;
    return C_EXCESS * excess / n + C_DISJOINT * disjoint / n + C_WEIGHT * avgWeightDiff;
}

void Population::mutateWeights(Genome& g, bool explore) {
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

    for (auto& s : species) {
        std::sort(s.begin(), s.end(), [&](int a, int b) {
            return m_genomes[a].fitness > m_genomes[b].fitness;
        });
    }

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
    size_t bestSpecies = 0;
    for (size_t si = 1; si < species.size(); ++si) {
        if (speciesAdj[si] > speciesAdj[bestSpecies]) bestSpecies = si;
    }
    allotted[bestSpecies] += popSize - assigned;

    std::vector<Genome> next;
    next.reserve(popSize);
    for (size_t si = 0; si < species.size(); ++si) {
        auto const& members = species[si];
        int quota = allotted[si];
        if (quota <= 0) continue;

        if (members.size() >= 5) {
            next.push_back(m_genomes[members.front()]);
            --quota;
        }

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
    while (static_cast<int>(next.size()) < popSize) {
        Genome child = best();
        mutate(child, explore);
        next.push_back(std::move(child));
    }
    next.resize(popSize);

    if (explore) {
        int const fresh = std::max(1, popSize / 5);
        for (int i = popSize - fresh; i < popSize; ++i) {
            next[i] = freshGenome();
        }
    }

    if (elite && elite->nodeCount > 0) {
        next.back() = *elite;
        next.back().fitness = 0.0;
    }

    m_genomes = std::move(next);
}

namespace {

constexpr uint32_t MAX_CONNS = 1'000'000;
constexpr uint32_t MAX_NODES = 1'000'000;
constexpr uint32_t MAX_GENOMES = 100'000;
constexpr uint32_t MAX_INNOVATIONS = 10'000'000;
constexpr uint32_t MAX_TAPE = 10'000'000;
constexpr uint32_t MAX_RNG = 100'000;

template <typename T>
void writePod(std::ostream& out, T const& value) {
    out.write(reinterpret_cast<char const*>(&value), sizeof(T));
}

template <typename T>
bool readPod(std::istream& in, T& out) {
    in.read(reinterpret_cast<char*>(&out), sizeof(T));
    return in.good();
}

}

void writeGenome(std::ostream& out, Genome const& g) {
    writePod(out, g.numInputs);
    writePod(out, g.numOutputs);
    writePod(out, g.nodeCount);
    writePod(out, g.fitness);
    writePod(out, g.reachStep);

    writePod(out, static_cast<uint32_t>(g.depths.size()));
    if (!g.depths.empty()) {
        out.write(reinterpret_cast<char const*>(g.depths.data()),
                  static_cast<std::streamsize>(g.depths.size())
                      * sizeof(double));
    }

    writePod(out, static_cast<uint32_t>(g.conns.size()));
    for (auto const& c : g.conns) {
        writePod(out, c.in);
        writePod(out, c.out);
        writePod(out, c.weight);
        writePod(out, static_cast<uint8_t>(c.enabled ? 1 : 0));
        writePod(out, c.innovation);
    }

    writePod(out, static_cast<uint32_t>(g.tapeToggles.size()));
    if (!g.tapeToggles.empty()) {
        out.write(reinterpret_cast<char const*>(g.tapeToggles.data()),
                  static_cast<std::streamsize>(g.tapeToggles.size())
                      * sizeof(int));
    }
}

bool readGenome(std::istream& in, Genome& g) {
    if (!readPod(in, g.numInputs) || !readPod(in, g.numOutputs)
        || !readPod(in, g.nodeCount) || !readPod(in, g.fitness)
        || !readPod(in, g.reachStep)) {
        return false;
    }
    if (g.nodeCount < 0 || static_cast<uint32_t>(g.nodeCount) > MAX_NODES) {
        return false;
    }

    uint32_t depthCount = 0;
    if (!readPod(in, depthCount) || depthCount > MAX_NODES) return false;
    g.depths.resize(depthCount);
    if (depthCount > 0) {
        in.read(reinterpret_cast<char*>(g.depths.data()),
                static_cast<std::streamsize>(depthCount) * sizeof(double));
        if (!in) return false;
    }

    uint32_t connCount = 0;
    if (!readPod(in, connCount) || connCount > MAX_CONNS) return false;
    g.conns.resize(connCount);
    for (auto& c : g.conns) {
        uint8_t enabled = 0;
        if (!readPod(in, c.in) || !readPod(in, c.out) || !readPod(in, c.weight)
            || !readPod(in, enabled) || !readPod(in, c.innovation)) {
            return false;
        }
        c.enabled = enabled != 0;
    }

    uint32_t tapeCount = 0;
    if (!readPod(in, tapeCount) || tapeCount > MAX_TAPE) return false;
    g.tapeToggles.resize(tapeCount);
    if (tapeCount > 0) {
        in.read(reinterpret_cast<char*>(g.tapeToggles.data()),
                static_cast<std::streamsize>(tapeCount) * sizeof(int));
        if (!in) return false;
    }
    return true;
}

void Population::writeState(std::ostream& out) const {
    writePod(out, m_nextInnovation);

    writePod(out, static_cast<uint32_t>(m_innovations.size()));
    for (auto const& [key, value] : m_innovations) {
        writePod(out, key);
        writePod(out, value);
    }

    std::ostringstream rng;
    rng << m_rng;
    std::string const rngState = rng.str();
    writePod(out, static_cast<uint32_t>(rngState.size()));
    if (!rngState.empty()) {
        out.write(rngState.data(),
                  static_cast<std::streamsize>(rngState.size()));
    }

    writeGenome(out, m_proto);

    writePod(out, static_cast<uint32_t>(m_genomes.size()));
    for (auto const& g : m_genomes) {
        writeGenome(out, g);
    }
}

bool Population::readState(std::istream& in) {
    if (!readPod(in, m_nextInnovation)) return false;

    uint32_t innovCount = 0;
    if (!readPod(in, innovCount) || innovCount > MAX_INNOVATIONS) return false;
    m_innovations.clear();
    m_innovations.reserve(innovCount);
    for (uint32_t i = 0; i < innovCount; ++i) {
        int64_t key = 0;
        int value = 0;
        if (!readPod(in, key) || !readPod(in, value)) return false;
        m_innovations.emplace(key, value);
    }

    uint32_t rngLen = 0;
    if (!readPod(in, rngLen) || rngLen > MAX_RNG) return false;
    std::string rngState(rngLen, '\0');
    if (rngLen > 0) {
        in.read(rngState.data(), static_cast<std::streamsize>(rngLen));
        if (!in) return false;
    }
    std::istringstream rng(rngState);
    rng >> m_rng;
    if (!rng) return false;

    if (!readGenome(in, m_proto)) return false;

    uint32_t genomeCount = 0;
    if (!readPod(in, genomeCount) || genomeCount > MAX_GENOMES) return false;
    m_genomes.clear();
    m_genomes.resize(genomeCount);
    for (auto& g : m_genomes) {
        if (!readGenome(in, g)) return false;
    }
    return true;
}

}
