#include "server-slot-lora.h"

#include "json.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace {

using json = common_json;

json state_signature(const std::string & state_path) {
    return json{
        {"file_size", (uint64_t) std::filesystem::file_size(state_path)},
        {"file_time", (int64_t) std::filesystem::last_write_time(state_path).time_since_epoch().count()},
    };
}

} // namespace

void server_slot_lora_save(const std::string & state_path, const std::vector<common_adapter_lora_info> & adapters) {
    json record = state_signature(state_path);
    record["version"] = 1;
    record["adapters"] = json::array();
    for (const auto & adapter : adapters) {
        const json file = state_signature(adapter.path);
        record["adapters"].push_back(json{
            {"path", adapter.path},
            {"scale", adapter.scale},
            {"file_size", file.at("file_size")},
            {"file_time", file.at("file_time")},
        });
    }

    const std::string destination = state_path + ".lora.json";
    const std::string temporary = destination + ".tmp";
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file || !(file << record.dump()) || !file.flush()) {
            throw std::runtime_error("cannot save slot adapter metadata");
        }
    }
    std::filesystem::rename(temporary, destination);
}

std::vector<common_adapter_lora_info> server_slot_lora_load(
    const std::string & state_path, const std::vector<common_adapter_lora_info> & loaded) {
    std::ifstream file(state_path + ".lora.json", std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0 || file.tellg() > 64 * 1024) {
        throw std::runtime_error("missing or invalid slot adapter metadata");
    }
    std::string contents((size_t) file.tellg(), '\0');
    file.seekg(0);
    if (!file.read(contents.data(), contents.size())) {
        throw std::runtime_error("cannot read slot adapter metadata");
    }
    const json record = json::parse(contents);
    const json signature = state_signature(state_path);
    if (record.at("version") != 1 || record.at("file_size") != signature.at("file_size") ||
        record.at("file_time") != signature.at("file_time")) {
        throw std::runtime_error("slot adapter metadata does not match the saved KV state");
    }
    const auto & entries = record.at("adapters");
    if (!entries.is_array() || entries.size() != loaded.size()) {
        throw std::runtime_error("slot adapter set differs from loaded adapters");
    }
    auto result = loaded;
    for (size_t i = 0; i < loaded.size(); i++) {
        const auto & entry = entries.at(i);
        const float scale = entry.at("scale").get<float>();
        const json file = state_signature(loaded[i].path);
        if (entry.at("path") != loaded[i].path || !std::isfinite(scale) ||
            entry.at("file_size") != file.at("file_size") || entry.at("file_time") != file.at("file_time")) {
            throw std::runtime_error("slot was saved with an incompatible adapter");
        }
        result[i].scale = scale;
    }
    return result;
}
