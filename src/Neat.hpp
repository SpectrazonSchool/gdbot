#pragma once

#include <cstdint>
#include <iosfwd>
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

struct Genome {
    int numInputs = 0;
    int numOutputs = 0;
    int nodeCount = 0;
    std::vector<double> depths;
    std::vector<ConnGene> conns;
    double fitness = 0.0;

    std::vector<int> tapeToggles;
    int reachStep = 0;

    int outputId(int i) const { return numInputs + 1 + i; }
};

void writeGenome(std::ostream& out, Genome const& g);
bool readGenome(std::istream& in, Genome& g);

class Network {
public:
    explicit Network(Genome const& genome);

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
    std::vector<Node> m_order;
    std::vector<double> m_values;
    int m_outputId;
};

class Population {
public:
    Population(int size, int numInputs, int numOutputs, uint32_t seed);

    int size() const { return static_cast<int>(m_genomes.size()); }
    Genome const& genome(int i) const { return m_genomes[i]; }
    void setFitness(int i, double fitness) { m_genomes[i].fitness = fitness; }
    void setTape(int i, std::vector<int> toggles, int reachStep) {
        m_genomes[i].tapeToggles = std::move(toggles);
        m_genomes[i].reachStep = reachStep;
    }

    Genome const& best() const;

    void epoch(bool explore = false, Genome const* elite = nullptr);

    void writeState(std::ostream& out) const;
    bool readState(std::istream& in);

private:
    int innovationFor(int in, int out);
    void randomizeWeights(Genome& g);
    Genome freshGenome();
    void mutate(Genome& g, bool explore);
    void mutateWeights(Genome& g, bool explore);
    bool mutateAddConnection(Genome& g);
    void mutateAddNode(Genome& g);
    Genome crossover(Genome const& fitter, Genome const& other);
    static double compatibility(Genome const& a, Genome const& b);
    static void pruneDisabled(Genome& g);

    double rand01() { return m_uniform(m_rng); }
    double randWeight() { return m_uniform(m_rng) * 4.0 - 2.0; }

    std::vector<Genome> m_genomes;
    Genome m_proto;
    std::mt19937 m_rng;
    std::uniform_real_distribution<double> m_uniform{0.0, 1.0};
    std::normal_distribution<double> m_gauss{0.0, 0.5};
    std::unordered_map<int64_t, int> m_innovations;
    int m_nextInnovation = 0;
};

}
