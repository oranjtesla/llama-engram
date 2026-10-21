#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

class server_jev_head {
public:
    static server_jev_head load(const std::string & directory);

    std::vector<float> score(const std::string & kind, const float * hidden,
                             size_t n_hidden, size_t n_outputs) const;
    size_t max_options(const std::string & kind) const;
    size_t hidden_size() const { return n_hidden; }
    float temperature(const std::string & kind) const;

private:
    size_t n_hidden = 0;
    size_t n_slots = 0;
    float softcap = 0.0f;
    std::map<std::string, std::pair<size_t, size_t>> ranges;
    std::map<std::string, float> temperatures;
    std::vector<float> weight;
    std::vector<float> bias;
};
