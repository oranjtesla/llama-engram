#include "chat.h"
#include "json.h"
#include "jev-bare-v1.h"
#include "jinja/runtime.h"

#include <iostream>
#include <stdexcept>
#include <string>

using json = common_json;

static std::string render(const json & input) {
    common_chat_template tmpl(JEV_BARE_V1_TEMPLATE, "", "");
    jinja::context ctx(tmpl.source());
    jinja::global_from_json(ctx, input, false);
    jinja::runtime runtime(ctx);
    return jinja::runtime::gather_string_parts(runtime.execute(tmpl.prog))->as_string().str();
}

int main() {
    try {
        const json bare = {
            {"type", "noul"}, {"state", "state"}, {"instructions", "question"},
            {"options", json::array({{{"key", "false"}, {"description", nullptr}},
                                      {{"key", "true"}, {"description", nullptr}}})}
        };
        if (render(bare) != "[kind] noul\n[state] state\n[question] question\n[options]\nfalse\ntrue\n[decision]:") {
            throw std::runtime_error("JEV plain Noul prompt does not match bare-v1");
        }
        json complex = bare;
        complex["state"] = {{"time", "morning"}, {"events", json::array({"test", "transfer"})}};
        complex["instructions"] = {{"question", "Has homeroom ended?"}};
        complex["options"][1]["description"] = {{"meaning", "Exam begun"}};
        const auto structured = render(complex);
        if (structured.find("\"events\":[\"test\",\"transfer\"]") == std::string::npos ||
            structured.find("\"question\":\"Has homeroom ended?\"") == std::string::npos ||
            structured.find("true: {\"meaning\":\"Exam begun\"}") == std::string::npos ||
            structured.find("[options]\nfalse\ntrue\n[decision]:") == std::string::npos) {
            throw std::runtime_error("JEV structured fields were not preserved by template");
        }
        complex["options"][1]["description"] = {{"meaning", "Exam not begun"}};
        if (render(complex) == structured) {
            throw std::runtime_error("JEV template silently ignored a criteria change");
        }
        json choice = bare;
        choice["type"] = "choice";
        choice["options"] = json::array({{{"key", "classroom"}, {"description", "school"}},
                                          {{"key", "hallway"}, {"description", json{{"place", "corridor"}}}}});
        if (render(choice).find("A) classroom: school\nB) hallway: {\"place\":\"corridor\"}") == std::string::npos) {
            throw std::runtime_error("JEV choice option rendering is incorrect");
        }
        json score = bare;
        score["type"] = "score";
        score["options"] = json::array();
        for (int i = 0; i < 6; i++) {
            score["options"].push_back(json{{"key", std::to_string(i)}, {"description", "level " + std::to_string(i)}});
        }
        const auto rubric = render(score);
        if (rubric.find("[criteria]\n0: level 0") == std::string::npos ||
            rubric.find("[options]\n0\n1\n2\n3\n4\n5\n[decision]:") == std::string::npos) {
            throw std::runtime_error("JEV native Score rubric or labels were omitted");
        }
        std::cout << "JEV Noul, Choice, Score and structured Jinja tests passed\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
