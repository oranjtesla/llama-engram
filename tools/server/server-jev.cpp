#include "server-jev.h"

#include "json.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = common_json;

std::vector<char> read_file(const std::filesystem::path & path, size_t max_size) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open JEV artifact: " + path.string());
    }
    const auto size = input.tellg();
    if (size < 0 || (uint64_t) size > max_size) {
        throw std::runtime_error("invalid JEV artifact size: " + path.string());
    }
    std::vector<char> content((size_t) size);
    input.seekg(0);
    if (!input.read(content.data(), content.size())) {
        throw std::runtime_error("cannot read JEV artifact: " + path.string());
    }
    return content;
}

json read_json(const std::filesystem::path & path) {
    const auto content = read_file(path, 1024 * 1024);
    return json::parse(std::string(content.data(), content.size()));
}

std::vector<float> read_tensor(const json & header, const std::vector<char> & data,
                               size_t data_start, const std::string & name, const std::vector<size_t> & shape) {
    const auto & tensor = header.at(name);
    if (tensor.at("dtype") != "F32" || tensor.at("shape").get<std::vector<size_t>>() != shape) {
        throw std::runtime_error("invalid JEV tensor format: " + name);
    }
    const auto offsets = tensor.at("data_offsets").get<std::vector<size_t>>();
    size_t count = 1;
    for (const size_t dim : shape) {
        if (dim == 0 || count > (1024 * 1024) / dim) {
            throw std::runtime_error("invalid JEV tensor dimensions: " + name);
        }
        count *= dim;
    }
    if (offsets.size() != 2 || offsets[0] > offsets[1] || offsets[1] > data.size() - data_start ||
        offsets[1] - offsets[0] != count * sizeof(float)) {
        throw std::runtime_error("invalid JEV tensor offsets: " + name);
    }
    std::vector<float> values(count);
    std::memcpy(values.data(), data.data() + data_start + offsets[0], count * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("nonfinite JEV tensor: " + name);
        }
    }
    return values;
}

} // namespace

server_jev_head server_jev_head::load(const std::string & directory) {
    const std::filesystem::path root(directory);
    const json config = read_json(root / "judge_config.json");
    const json calibration = read_json(root / "calibration_gold.json");

    server_jev_head result;
    result.n_hidden = config.at("hidden_size").get<size_t>();
    result.n_slots = config.at("slots").at("num_slots").get<size_t>();
    const auto range = config.at("slots").at("ranges").at("noul").get<std::vector<size_t>>();
    result.softcap = config.at("softcap").get<float>();
    result.temp_noul = calibration.at("per_kind").at("noul").get<float>();

    if (result.n_hidden != 2816 || result.n_slots != 24 || range.size() != 2 || range[1] != range[0] + 2 ||
        range[1] > result.n_slots || config.at("slots").at("template_version") != "bare-v1" ||
        config.at("readout").at("type") != "last_token" ||
        !std::isfinite(result.softcap) || result.softcap <= 0 ||
        !std::isfinite(result.temp_noul) || result.temp_noul <= 0) {
        throw std::runtime_error("unsupported JEV classifier configuration");
    }
    result.noul_first = range[0];

    const auto data = read_file(root / "head.safetensors", 4 * 1024 * 1024);
    if (data.size() < 8) {
        throw std::runtime_error("invalid JEV safetensors header");
    }
    uint64_t header_size = 0;
    for (size_t i = 0; i < 8; i++) {
        header_size |= (uint64_t) (uint8_t) data[i] << (8 * i);
    }
    if (header_size > data.size() - 8 || header_size > 1024 * 1024) {
        throw std::runtime_error("invalid JEV safetensors header length");
    }
    const json header = json::parse(std::string(data.data() + 8, header_size));
    result.weight = read_tensor(header, data, 8 + header_size, "proj.weight", {result.n_slots, result.n_hidden});
    result.bias = read_tensor(header, data, 8 + header_size, "proj.bias", {result.n_slots});
    return result;
}

std::string server_jev_head::render_noul(const std::string & state, const std::string & question) {
    return "[kind] noul\n[state] " + state + "\n[question] " + question +
           "\n[options]\nfalse\ntrue\n[decision]:";
}

std::array<float, 2> server_jev_head::score_noul(const float * hidden, size_t n_hidden_input) const {
    if (hidden == nullptr || n_hidden_input != n_hidden) {
        throw std::invalid_argument("JEV hidden state has wrong dimensions");
    }
    std::array<float, 2> scores;
    for (size_t i = 0; i < scores.size(); i++) {
        const size_t slot = noul_first + i;
        float score = bias[slot];
        for (size_t j = 0; j < n_hidden; j++) {
            if (!std::isfinite(hidden[j])) {
                throw std::runtime_error("JEV hidden state contains nonfinite values");
            }
            score += weight[slot * n_hidden + j] * hidden[j];
        }
        scores[i] = softcap * std::tanh(score / softcap);
    }
    return scores;
}
