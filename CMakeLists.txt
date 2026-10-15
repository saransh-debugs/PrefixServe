cmake_minimum_required(VERSION 3.20)
project(prefixserve LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # core is linked into a Python .so
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)    # for clangd

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(PREFIXSERVE_BUILD_TESTS "Build C++ tests and micro-benchmark" OFF)
option(PREFIXSERVE_SANITIZE "Build test_core with ASan + UBSan" ON)

set(CORE_SOURCES csrc/block_manager.cpp csrc/radix_tree.cpp)
set(CORE_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Werror)

add_library(prefixserve_core STATIC ${CORE_SOURCES})
target_include_directories(prefixserve_core PUBLIC csrc)
target_compile_options(prefixserve_core PRIVATE ${CORE_WARNINGS})

# ---- Python module (built by scikit-build-core via `pip install .`) ----
if(SKBUILD)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core MODULE csrc/bindings.cpp)
  target_link_libraries(_core PRIVATE prefixserve_core)
  install(TARGETS _core LIBRARY DESTINATION prefixserve)
endif()

# ---- C++ tests ----
if(PREFIXSERVE_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()

  # Core sources are compiled into each test directly, so the sanitizer
  # flags apply to the core code too, not just the test file.
  add_executable(test_core tests/cpp/test_core.cpp ${CORE_SOURCES})
  target_include_directories(test_core PRIVATE csrc)
  target_compile_options(test_core PRIVATE ${CORE_WARNINGS} -Wno-sign-conversion -Wno-conversion)
  target_link_libraries(test_core PRIVATE GTest::gtest_main)
  if(PREFIXSERVE_SANITIZE)
    target_compile_options(test_core PRIVATE
      -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
    target_link_options(test_core PRIVATE -fsanitize=address,undefined)
  endif()
  add_test(NAME test_core COMMAND test_core)

  # No sanitizers here: this test replaces global operator new to count
  # allocations, and ASan wants to own operator new itself.
  add_executable(test_no_alloc tests/cpp/test_no_alloc.cpp ${CORE_SOURCES})
  target_include_directories(test_no_alloc PRIVATE csrc)
  target_link_libraries(test_no_alloc PRIVATE GTest::gtest_main)
  add_test(NAME test_no_alloc COMMAND test_no_alloc)

  add_executable(micro_core bench/micro_core.cpp ${CORE_SOURCES})
  target_include_directories(micro_core PRIVATE csrc)
  target_compile_options(micro_core PRIVATE -O3)
endif()
