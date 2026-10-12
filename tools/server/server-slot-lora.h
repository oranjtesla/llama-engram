#pragma once

#include "common.h"

#include <string>
#include <vector>

void server_slot_lora_save(const std::string & state_path, const std::vector<common_adapter_lora_info> & adapters);

std::vector<common_adapter_lora_info> server_slot_lora_load(
    const std::string & state_path, const std::vector<common_adapter_lora_info> & loaded);
