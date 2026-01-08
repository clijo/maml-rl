#include "MarsLander.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>

// Constants
const int POPULATION_SIZE = 100;
const int GENOME_LENGTH = 100; // Frames
const int GENERATIONS = 500; // Increased from 50 to 500
const float ELITISM_RATE = 0.1f;
const float MUTATION_RATE = 0.2f;
const float MUTATION_STRENGTH = 0.5f;

// Gene: Action pair {rotate, power}
struct Gene {
    float rotate; // Normalized [-1, 1] -> [-15, 15] delta
    float power;  // Normalized [-1, 1] -> [-1, 1] delta
};

struct Individual {
    std::vector<Gene> genome;
    float fitness;
};

// Fitness Function (Replicated from Python logic)
float evaluate(Individual& ind, int seed, bool debug = false) {
    MarsLander env;
    std::map<std::string, float> task_params;
    // Task: Fixed Start and Target (from Python config)
    // Target X: Random in [500, 6500] -> Let's test a specific hard case
    // For GA validation, let's pick a fixed target to prove it learns.
    // Say Target X = 4500 (Start 3500) -> Needs lateral move.
    task_params["target_x"] = 4500.0f;
    task_params["target_y"] = 100.0f;
    task_params["landing_width"] = 1000.0f;
    task_params["start_x"] = 3500.0f;
    task_params["start_y"] = 2500.0f;
    
    env.reset(seed, task_params);
    
    float total_reward = 0.0f;
    float prev_potential = 0.0f;
    
    // Init Potential
    {
        float dx = std::abs(env.state.x - env.task.target_x);
        float dy = std::abs(env.state.y - env.task.target_y);
        float dist_norm = std::sqrt(dx*dx + dy*dy) / 3000.0f;
        prev_potential = - (1.0f * dist_norm);
    }
    
    for (int i = 0; i < GENOME_LENGTH; ++i) {
        std::vector<float> action = {ind.genome[i].rotate, ind.genome[i].power};
        auto result = env.step(action);
        bool done = std::get<2>(result);
        float reward = 0.0f;
        
        // --- Shaping Logic ---
        float dx = std::abs(env.state.x - env.task.target_x);
        float dy = std::abs(env.state.y - env.task.target_y);
        float dist_norm = std::sqrt(dx*dx + dy*dy) / 3000.0f;
        float vel_norm = std::sqrt(env.state.hs*env.state.hs + env.state.vs*env.state.vs) / 100.0f;
        
        float potential = - (1.0f * dist_norm + 0.1f * vel_norm);
        
        reward += 10.0f * (potential - prev_potential);
        prev_potential = potential;
        
        if (done) {
            float cpp_reward = std::get<1>(result);
            if (cpp_reward > 0.5f) {
                reward += 1000.0f; // SUCCESS
                if (debug) std::cout << "LANDED!" << std::endl;
            } else {
                reward -= 100.0f; // CRASH
                // Crash Penalties
                reward -= 1.0f * std::abs(env.state.hs);
                reward -= 1.0f * std::abs(env.state.vs);
            }
            total_reward += reward;
            break;
        }
        
        total_reward += reward;
    }
    
    return total_reward;
}

// GA Solver
int main() {
    std::srand(std::time(0));
    std::default_random_engine gen(std::time(0));
    std::uniform_real_distribution<float> act_dist(-1.0f, 1.0f);
    
    // Init Population
    std::vector<Individual> population(POPULATION_SIZE);
    for (auto& ind : population) {
        ind.genome.resize(GENOME_LENGTH);
        for (auto& gene : ind.genome) {
            gene.rotate = act_dist(gen);
            gene.power = act_dist(gen); // Start with random thrust
        }
    }
    
    std::cout << "Starting GA Evolution for Mars Lander..." << std::endl;
    std::cout << "Target: X=4500 (Start X=3500). Needs lateral flight." << std::endl;
    
    int seed = 42;
    
    for (int generation = 0; generation < GENERATIONS; ++generation) {
        for (auto& ind : population) {
            ind.fitness = evaluate(ind, seed); // Fixed seed for deterministic env during eval
        }
        
        // Sort
        std::sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
            return a.fitness > b.fitness;
        });
        
        // Print Best
        std::cout << "Gen " << generation << " | Best Fitness: " << population[0].fitness << std::endl;
        
        if (generation == GENERATIONS - 1) break;
        
        // Elitism
        std::vector<Individual> next_gen;
        int elite_count = (int)(POPULATION_SIZE * ELITISM_RATE);
        for (int i = 0; i < elite_count; ++i) {
            next_gen.push_back(population[i]);
        }
        
        // Selection & Crossover
        while (next_gen.size() < POPULATION_SIZE) {
            // Tournament
            int i1 = std::rand() % (POPULATION_SIZE / 2); // Pick from top 50%
            int i2 = std::rand() % (POPULATION_SIZE / 2);
            const auto& p1 = population[i1];
            const auto& p2 = population[i2];
            
            // Crossover
            Individual child;
            child.genome.resize(GENOME_LENGTH);
            int split = std::rand() % GENOME_LENGTH;
            for (int k = 0; k < GENOME_LENGTH; ++k) {
                if (k < split) child.genome[k] = p1.genome[k];
                else child.genome[k] = p2.genome[k];
            }
            
            // Mutation
            for (auto& gene : child.genome) {
                if ((float)std::rand() / RAND_MAX < MUTATION_RATE) {
                    gene.rotate += ((float)std::rand() / RAND_MAX - 0.5f) * MUTATION_STRENGTH;
                    gene.power += ((float)std::rand() / RAND_MAX - 0.5f) * MUTATION_STRENGTH;
                    // Clamp
                    gene.rotate = std::max(-1.0f, std::min(1.0f, gene.rotate));
                    gene.power = std::max(-1.0f, std::min(1.0f, gene.power));
                }
            }
            next_gen.push_back(child);
        }
        population = next_gen;
    }
    
    // Final Run validation
    std::cout << "--- Best Run Simulation ---" << std::endl;
    evaluate(population[0], seed, true);
    
    return 0;
}
