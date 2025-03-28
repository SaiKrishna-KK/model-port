/*
 * cpp_inference.cpp - Example of running inference with TVM compiled models in C++
 * 
 * This example demonstrates how to load and run a model compiled with ModelPort
 * directly from C++ without Python dependencies.
 * 
 * Compilation:
 *   g++ -std=c++14 -O2 -o cpp_inference cpp_inference.cpp -ltvm_runtime -ldl
 * 
 * Usage:
 *   ./cpp_inference model_graph.json model_lib.so model_params.params
 */

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <random>

// Helper function to read parameters from file
std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl;
        exit(1);
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::string buffer(size, ' ');
    if (!file.read(&buffer[0], size)) {
        std::cerr << "Error reading file: " << path << std::endl;
        exit(1);
    }
    
    return buffer;
}

// Simple random data generator for testing
template<typename T>
std::vector<T> generate_random_data(const std::vector<int64_t>& shape) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    size_t num_elements = 1;
    for (auto dim : shape) {
        num_elements *= dim;
    }
    
    std::vector<T> data(num_elements);
    for (size_t i = 0; i < num_elements; i++) {
        data[i] = static_cast<T>(dis(gen));
    }
    
    return data;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_graph.json> <model_lib.so> <model_params.params>" << std::endl;
        return 1;
    }
    
    // Parse command line arguments
    std::string graph_path = argv[1];
    std::string lib_path = argv[2];
    std::string params_path = argv[3];
    
    std::cout << "Loading model from:" << std::endl;
    std::cout << "  Graph: " << graph_path << std::endl;
    std::cout << "  Library: " << lib_path << std::endl;
    std::cout << "  Parameters: " << params_path << std::endl;
    
    try {
        // Load the model library
        tvm::runtime::Module mod_lib = tvm::runtime::Module::LoadFromFile(lib_path);
        
        // Read graph JSON
        std::string graph_json = read_file(graph_path);
        
        // Read parameters
        std::string params_blob = read_file(params_path);
        
        // Create TVM runtime module
        tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_executor.create"))(
            graph_json, mod_lib, 0, 0);
        
        // Get runtime module functions
        tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
        tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
        tvm::runtime::PackedFunc run = mod.GetFunction("run");
        tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
        tvm::runtime::PackedFunc get_num_outputs = mod.GetFunction("get_num_outputs");
        tvm::runtime::PackedFunc get_input = mod.GetFunction("get_input");
        tvm::runtime::PackedFunc get_num_inputs = mod.GetFunction("get_num_inputs");
        
        // Load parameters
        load_params(params_blob);
        
        // Get number of inputs
        int num_inputs = static_cast<int>(get_num_inputs());
        std::cout << "Model has " << num_inputs << " input(s)" << std::endl;
        
        // Process each input
        for (int i = 0; i < num_inputs; i++) {
            // Get input name
            std::string input_name = mod.GetFunction("get_input_name")(i);
            std::cout << "Input " << i << ": " << input_name << std::endl;
            
            // For this example, let's create a simple dummy input
            std::vector<int64_t> input_shape = {1, 3, 224, 224}; // Default for many vision models
            std::cout << "  Using default shape: [1, 3, 224, 224]" << std::endl;
            
            // Generate random float32 data
            auto input_data = generate_random_data<float>(input_shape);
            std::cout << "  Generated random input data" << std::endl;
            
            // Create DLTensor for input
            DLTensor* input_tensor;
            TVMArrayAlloc(input_shape.data(), input_shape.size(), 
                         kDLFloat, 32, 1, kDLCPU, 0, &input_tensor);
            
            // Copy data to input tensor
            memcpy(input_tensor->data, input_data.data(), input_data.size() * sizeof(float));
            
            // Set the input
            if (input_name.empty()) {
                set_input(i, input_tensor);
            } else {
                set_input(input_name, input_tensor);
            }
            
            // Free the allocated tensor
            TVMArrayFree(input_tensor);
        }
        
        // Run inference (time it)
        std::cout << "Running inference..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        run();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Inference took " << elapsed.count() * 1000 << " ms" << std::endl;
        
        // Get outputs
        int num_outputs = static_cast<int>(get_num_outputs());
        std::cout << "Model has " << num_outputs << " output(s)" << std::endl;
        
        for (int i = 0; i < num_outputs; i++) {
            // Get output tensor
            tvm::runtime::NDArray output = get_output(i);
            
            // Get shape information
            std::vector<int64_t> output_shape;
            for (int j = 0; j < output->ndim; j++) {
                output_shape.push_back(output->shape[j]);
            }
            
            // Print shape
            std::cout << "Output " << i << " shape: [";
            for (size_t j = 0; j < output_shape.size(); j++) {
                std::cout << output_shape[j];
                if (j < output_shape.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;
            
            // Calculate number of elements
            size_t num_elements = 1;
            for (auto dim : output_shape) {
                num_elements *= dim;
            }
            
            // Optional: Print first few values
            const int max_display = 5;
            float* data_ptr = static_cast<float*>(output->data);
            
            std::cout << "  First " << std::min(max_display, static_cast<int>(num_elements)) 
                      << " values: ";
            for (int j = 0; j < std::min(max_display, static_cast<int>(num_elements)); j++) {
                std::cout << data_ptr[j] << " ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "Inference completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 