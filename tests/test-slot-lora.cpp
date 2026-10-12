#include "server-slot-lora.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    const std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("test-slot-lora-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        {
            std::ofstream file(path, std::ios::binary);
            file << "slot state";
        }
        const std::string adapter_path = path.string() + ".gguf";
        {
            std::ofstream file(adapter_path);
            file << "adapter";
        }
        std::vector<common_adapter_lora_info> adapters = {{adapter_path, 1.0f, "", "", nullptr}};
        server_slot_lora_save(path.string(), adapters);
        if (server_slot_lora_load(path.string(), adapters)[0].scale != 1.0f) {
            throw std::runtime_error("adapter scale did not survive slot restore");
        }

        adapters[0].path = path.string();
        try {
            server_slot_lora_load(path.string(), adapters);
            throw std::runtime_error("restore accepted incompatible adapter identity");
        } catch (const std::runtime_error & error) {
            if (std::string(error.what()) == "restore accepted incompatible adapter identity") {
                throw;
            }
        }

        adapters[0].path = adapter_path;
        {
            std::ofstream file(path, std::ios::binary | std::ios::app);
            file << "changed";
        }
        try {
            server_slot_lora_load(path.string(), adapters);
            throw std::runtime_error("restore accepted modified state file");
        } catch (const std::runtime_error & error) {
            if (std::string(error.what()) == "restore accepted modified state file") {
                throw;
            }
        }
    } catch (...) {
        std::filesystem::remove(path);
        std::filesystem::remove(path.string() + ".lora.json");
        std::filesystem::remove(path.string() + ".gguf");
        throw;
    }
    std::filesystem::remove(path);
    std::filesystem::remove(path.string() + ".lora.json");
    return 0;
}
