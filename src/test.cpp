#include <mujoco/mujoco.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "tests/test_base.h"
#include "tests/smooth_test.h"

// Factory function to create test instances
std::unique_ptr<TestBase> create_test(const std::string& test_name) {
    if (test_name == "kinematics") {
        return std::make_unique<SmoothTest>();
    }
    return nullptr;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <test_name> <xml_file> <batch_size>\n"
              << "Available tests:\n"
              << "  kinematics    - Test kinematics computation\n";
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::string test_name = argv[1];
    std::string xml_file = argv[2];
    int batch_size = std::atoi(argv[3]);

    // Create test instance
    auto test = create_test(test_name);
    if (!test) {
        std::cerr << "Unknown test: " << test_name << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::cout << "Running " << test->get_name() << " test\n";
    std::cout << "Loading MuJoCo model from: " << xml_file << std::endl;
    std::cout << "Using batch_size = " << batch_size << std::endl;

    // Load model
    char error[1000];
    mjModel* m = mj_loadXML(xml_file.c_str(), 0, error, 1000);
    if (!m) {
        std::cerr << "Error loading model: " << error << std::endl;
        return EXIT_FAILURE;
    }

    mjData* d = mj_makeData(m);

    // Initialize and run test
    test->init(m, d, batch_size);
    bool passed = test->run_test();

    // Cleanup MuJoCo resources
    mj_deleteData(d);
    mj_deleteModel(m);

    if (passed) {
        std::cout << "Test passed!" << std::endl;
        return EXIT_SUCCESS;
    } else {
        std::cerr << "Test failed!" << std::endl;
        return EXIT_FAILURE;
    }
} 