#include "server-jev.h"

#include "json.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: test-jev-head JEV_DIRECTORY E1_EMBEDDING_JSON\n";
        return 2;
    }

    try {
        const server_jev_head head = server_jev_head::load(argv[1]);
        std::ifstream file(argv[2]);
        if (!file) {
            throw std::runtime_error("cannot open embedding fixture");
        }
        const std::string text(std::istreambuf_iterator<char>{file}, {});
        const common_json payload = common_json::parse(text);
        const auto h = payload.at(0).at("embedding").at(0).get<std::vector<float>>();
        const auto scores = head.score_noul(h.data(), h.size());
        const double p_true = 1.0 / (1.0 + std::exp((scores[0] - scores[1]) / head.temperature()));

        if (std::abs(p_true - 0.7435570359230042) > 1e-4) {
            throw std::runtime_error("JEV scoring differs from the reference result");
        }
        if (server_jev_head::render_noul("state", "question") !=
                "[kind] noul\n[state] state\n[question] question\n[options]\nfalse\ntrue\n[decision]:") {
            throw std::runtime_error("JEV prompt layout differs from bare-v1");
        }
        std::cout << "JEV classifier fixture passed: true = " << p_true << "\n";
    } catch (const std::exception & err) {
        std::cerr << err.what() << '\n';
        return 1;
    }
    return 0;
}
