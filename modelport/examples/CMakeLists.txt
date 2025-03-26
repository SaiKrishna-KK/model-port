cmake_minimum_required(VERSION 3.10)
project(modelport_cpp_examples)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find TVM package
find_package(TVM QUIET)
if(NOT TVM_FOUND)
    # Try to find TVM in common installation locations
    set(TVM_ROOT "$ENV{TVM_HOME}" CACHE PATH "Path to TVM installation")
    if(NOT TVM_ROOT)
        set(TVM_ROOT "/usr/local/tvm" CACHE PATH "Path to TVM installation")
    endif()
    
    # Check if TVM headers and libraries are available
    if(EXISTS "${TVM_ROOT}/include/tvm/runtime/module.h")
        message(STATUS "Found TVM headers at ${TVM_ROOT}/include")
        include_directories(${TVM_ROOT}/include)
        include_directories(${TVM_ROOT}/3rdparty/dlpack/include)
        include_directories(${TVM_ROOT}/3rdparty/dmlc-core/include)
        
        if(EXISTS "${TVM_ROOT}/lib/libtvm_runtime.so" OR 
           EXISTS "${TVM_ROOT}/lib/libtvm_runtime.dylib" OR
           EXISTS "${TVM_ROOT}/build/libtvm_runtime.so" OR
           EXISTS "${TVM_ROOT}/build/libtvm_runtime.dylib")
            message(STATUS "Found TVM runtime library")
            if(EXISTS "${TVM_ROOT}/lib/libtvm_runtime.so" OR
               EXISTS "${TVM_ROOT}/lib/libtvm_runtime.dylib")
                link_directories(${TVM_ROOT}/lib)
            else()
                link_directories(${TVM_ROOT}/build)
            endif()
        else()
            message(WARNING "TVM runtime library not found, please build TVM or specify correct TVM_ROOT")
        endif()
    else()
        message(WARNING "TVM headers not found, please build TVM or specify correct TVM_ROOT")
    endif()
else()
    message(STATUS "Found TVM package")
endif()

# Add the C++ example executable
add_executable(cpp_inference cpp_inference.cpp)

# Link against TVM runtime
target_link_libraries(cpp_inference tvm_runtime dl pthread)

# Installation
install(TARGETS cpp_inference DESTINATION bin)

# Print help message
message(STATUS "")
message(STATUS "To build the C++ example, run:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake .. -DTVM_ROOT=/path/to/tvm")
message(STATUS "  make")
message(STATUS "")
message(STATUS "To run the example:")
message(STATUS "  ./cpp_inference /path/to/model_graph.json /path/to/model_lib.so /path/to/model_params.params")
message(STATUS "") 