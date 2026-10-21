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
    result.softcap = config.at("softcap").get<float>();
    if (result.n_hidden != 2816 || result.n_slots != 24 ||
        config.at("slots").at("template_version") != "bare-v1" ||
        config.at("readout").at("type") != "last_token" ||
        !std::isfinite(result.softcap) || result.softcap <= 0) {
        throw std::runtime_error("unsupported JEV classifier configuration");
    }
    size_t expected_first = 0;
    for (const auto & [kind, width] : std::vector<std::pair<std::string, size_t>>{{"noul", 2}, {"score", 6}, {"choice", 16}}) {
        const auto range = config.at("slots").at("ranges").at(kind).get<std::vector<size_t>>();
        const float temperature = calibration.at("per_kind").at(kind).get<float>();
        if (range.size() != 2 || range[0] != expected_first || range[1] != range[0] + width ||
            range[1] > result.n_slots || !std::isfinite(temperature) || temperature <= 0) {
            throw std::runtime_error("unsupported JEV output range or calibration: " + kind);
        }
        result.ranges[kind] = {range[0], range[1]};
        result.temperatures[kind] = temperature;
        expected_first = range[1];
    }
    if (expected_first != result.n_slots) {
        throw std::runtime_error("JEV classifier ranges do not cover head outputs");
    }

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

size_t server_jev_head::max_options(const std::string & kind) const {
    const auto & range = ranges.at(kind);
    return range.second - range.first;
}

float server_jev_head::temperature(const std::string & kind) const {
    return temperatures.at(kind);
}

std::vector<float> server_jev_head::score(
        const std::string & kind, const float * hidden, size_t n_hidden_input, size_t n_outputs) const {
    if (hidden == nullptr || n_hidden_input != n_hidden) {
        throw std::invalid_argument("JEV hidden state has wrong dimensions");
    }
    const size_t first = ranges.at(kind).first;
    if (n_outputs < 2 || n_outputs > max_options(kind)) {
        throw std::invalid_argument("JEV output count exceeds trained head capacity");
    }
    for (size_t j = 0; j < n_hidden; j++) {
        if (!std::isfinite(hidden[j])) {
            throw std::runtime_error("JEV hidden state contains nonfinite values");
        }
    }
    std::vector<float> scores(n_outputs);
    for (size_t i = 0; i < n_outputs; i++) {
        const size_t slot = first + i;
        float value = bias[slot];
        for (size_t j = 0; j < n_hidden; j++) {
            value += weight[slot * n_hidden + j] * hidden[j];
        }
        scores[i] = softcap * std::tanh(value / softcap);
    }
    return scores;
}
