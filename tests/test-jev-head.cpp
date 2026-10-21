#include "server-jev.h"

#include "json.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static void check_distribution(const server_jev_head & head, const std::string & kind,
                               const std::vector<float> & hidden, const std::vector<double> & expected) {
    const auto logits = head.score(kind, hidden.data(), hidden.size(), expected.size());
    const float maximum = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    std::vector<double> probabilities;
    for (const float logit : logits) {
        probabilities.push_back(std::exp((logit - maximum) / head.temperature(kind)));
        sum += probabilities.back();
    }
    for (size_t i = 0; i < expected.size(); i++) {
        if (std::abs(probabilities[i] / sum - expected[i]) > 1e-4) {
            throw std::runtime_error(kind + " classifier scores differ from Python reference");
        }
    }
}

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
        const std::string content(std::istreambuf_iterator<char>{file}, {});
        const common_json payload = common_json::parse(content);
        const auto hidden = payload.at(0).at("embedding").at(0).get<std::vector<float>>();
        check_distribution(head, "noul", hidden, {0.2564430, 0.7435570});
        check_distribution(head, "score", hidden,
            {0.06058596, 0.26436827, 0.12329444, 0.41657734, 0.08627111, 0.04890283});
        check_distribution(head, "choice", hidden,
            {0.29678378, 0.29852158, 0.06037169, 0.07282730, 0.01616573, 0.03597815,
             0.02205181, 0.01196057, 0.01136802, 0.01735319, 0.03275004, 0.04204860,
             0.01118708, 0.02610767, 0.02060699, 0.02391775});
        std::cout << "JEV noul/score/choice reference scores passed\n";
    } catch (const std::exception & err) {
        std::cerr << err.what() << '\n';
        return 1;
    }
    return 0;
}
