#include "MarsLander.hpp"
#include <algorithm>
#include <iostream>

// Pi constant
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

MarsLander::MarsLander() : rng(42) {}

std::vector<float> MarsLander::reset(unsigned int seed, const std::map<std::string, float>& task_dict) {
    if (seed != 0) rng.seed(seed);
    
    // Parse Task
    if (task_dict.count("gravity")) task.gravity = task_dict.at("gravity");
    if (task_dict.count("wind_power")) task.wind_x = task_dict.at("wind_power"); // Simplified wind model
    if (task_dict.count("target_x")) task.target_x = task_dict.at("target_x");
    if (task_dict.count("target_y")) task.target_y = task_dict.at("target_y");
    if (task_dict.count("landing_width")) task.landing_width = task_dict.at("landing_width");
    if (task_dict.count("start_x")) task.start_x = task_dict.at("start_x");
    if (task_dict.count("start_y")) task.start_y = task_dict.at("start_y");
    
    // Initial State
    // Use task start or random around it? Python checked both.
    // Let's assume task dict provides exact or we randomize if not provided?
    // For MAML, usually fixed start per task OR random start.
    // We'll stick to logic:
    
    state.x = task.start_x;
    state.y = task.start_y;
    
    // Randomize initial velocity/angle slightly? Or strictly learned?
    // User asked for "fixed other parameters" in previous turn.
    // But reset usually adds some noise.
    std::uniform_real_distribution<float> dist_hs(-50.0f, 50.0f);
    std::uniform_real_distribution<float> dist_rot(-90.0f, 90.0f);
    std::uniform_real_distribution<float> dist_fuel(500.0f, 2000.0f);
    std::uniform_real_distribution<float> dist_vs(-50.0f, 0.0f);

    // If "start_hs" in task, use it.
    if (task_dict.count("start_hs")) state.hs = task_dict.at("start_hs");
    else state.hs = dist_hs(rng);
    
    if (task_dict.count("start_rotate")) state.rotate = task_dict.at("start_rotate");
    else state.rotate = dist_rot(rng);
    
    state.vs = dist_vs(rng);
    state.fuel = dist_fuel(rng);
    state.power = 0.0f;
    
    // prev_potential = compute_potential(state.x, state.y, state.hs, state.vs);
    
    return get_obs();
}

std::tuple<std::vector<float>, float, bool, bool, std::map<std::string, float>> MarsLander::step(std::vector<float> action) {
    // Action: [rotate_change, power_change] normalized [-1, 1]
    float rot_change = std::max(-1.0f, std::min(1.0f, action[0])) * 15.0f; // Scale to +/- 15 deg
    float pow_change = std::max(-1.0f, std::min(1.0f, action[1])); 

    // Update Control
    state.rotate = std::max(-90.0f, std::min(90.0f, state.rotate + rot_change));
    state.power = std::max(0.0f, std::min(4.0f, state.power + pow_change));
    
    // Fuel
    state.fuel -= state.power;
    if (state.fuel < 0) {
        state.fuel = 0;
        state.power = 0;
    }
    
    // Physics
    float rad = state.rotate * (M_PI / 180.0f);
    float thrust_mult = 8.0f; // Feasibility multiplier INCREASED to 8.0 (Super Arcade)
    
    float ax_thrust = -state.power * thrust_mult * std::sin(rad);
    float ay_thrust = state.power * thrust_mult * std::cos(rad);
    
    float ax = ax_thrust + task.wind_x;
    float ay = ay_thrust - task.gravity + task.wind_y;
    
    // Integration
    float x_new = state.x + state.hs + 0.5f * ax;
    float y_new = state.y + state.vs + 0.5f * ay;
    float hs_new = state.hs + ax;
    float vs_new = state.vs + ay;
    
    state.x = x_new;
    state.y = y_new;
    state.hs = hs_new;
    state.vs = vs_new;
    
    // Reward & Termination
    float reward = 0.0f;
    bool done = false;
    bool truncated = false; 
    
    // Potential Shaping REMOVED (Moved to Python)
    
    bool landed = false;
    bool crashed = false;
    
    // Check Platform Landing
    if (std::abs(state.y - task.target_y) < 10.0f) { 
        float dx = std::abs(state.x - task.target_x);
        if (dx < task.landing_width / 2.0f) {
            bool safe_speed = (std::abs(state.hs) < 20.0f) && (std::abs(state.vs) < 40.0f);
            bool safe_angle = (std::abs(state.rotate) < 10.0f);
            
            if (safe_speed && safe_angle) {
                landed = true;
            } else {
                crashed = true; 
            }
        }
    }
    
    if (state.y < 0.0f) {
        crashed = true;
    }
    if (state.x < 0.0f || state.x > MAP_WIDTH) {
        crashed = true;
    }
    
    // Sparse "True" Rewards
    if (landed) {
        reward = 1.0f; // Success
        done = true;
    } else if (crashed) {
        reward = 0.0f; // Fail
        done = true;
    }
    
    return {get_obs(), reward, done, truncated, {}};
}


std::vector<float> MarsLander::get_obs() {
    float dx = state.x - task.target_x;
    float dy = state.y - task.target_y; // Relative altitude to platform
    
    return {
        state.x / MAP_WIDTH,
        state.y / MAP_HEIGHT,
        state.hs / 100.0f,
        state.vs / 100.0f,
        state.rotate / 90.0f,
        state.power / 4.0f
    };
}


