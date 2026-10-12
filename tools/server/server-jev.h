#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

class server_jev_head {
public:
    static server_jev_head load(const std::string & directory);

    static std::string render_noul(const std::string & state, const std::string & question);

    std::array<float, 2> score_noul(const float * hidden, size_t n_hidden) const;

    size_t hidden_size() const { return n_hidden; }
    float temperature() const { return temp_noul; }

private:
    size_t n_hidden = 0;
    size_t n_slots = 0;
    size_t noul_first = 0;
    float softcap = 0.0f;
    float temp_noul = 1.0f;
    std::vector<float> weight;
    std::vector<float> bias;
};
