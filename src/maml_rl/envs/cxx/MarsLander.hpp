#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <tuple>
#include <map>
#include <string>

class MarsLander {
public:
    // State: x, y, hs, vs, fuel, rotate, power
    // Plus task params
    struct State {
        float x, y, hs, vs, fuel, rotate, power;
    };
    
    struct Task {
        float gravity = 3.711f;
        float wind_x = 0.0f;
        float wind_y = 0.0f;
        float target_x = 3500.0f;
        float target_y = 100.0f;
        float landing_width = 1000.0f;
        float start_x = 2500.0f;
        float start_y = 2500.0f;
    };

    MarsLander();
    
    // Core RL methods
    std::vector<float> reset(unsigned int seed, const std::map<std::string, float>& task_dict);
    std::tuple<std::vector<float>, float, bool, bool, std::map<std::string, float>> step(std::vector<float> action);
    
    // Getters
    std::vector<float> get_obs();
    
    // Public State for GA/Debugging
    State state;
    Task task;
    
private:
    std::mt19937 rng;
    
    // Physics Constants
    const float MAP_WIDTH = 7000.0f;
    const float MAP_HEIGHT = 3000.0f;
    
    // Helpers
    // float compute_potential(float x, float y, float hs, float vs);
    // float prev_potential;
    
};
