cmake_minimum_required(VERSION 3.20)
project(use_matrixGraph LANGUAGES CXX CUDA)

# Directory Structure
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(THIRD_PARTY_ROOT ${PROJECT_ROOT_DIR}/third_party)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_ROOT_DIR}/lib)
set(CMAKE_MODULE_PATH ${PROJECT_ROOT_DIR}/cmake)

cmake_policy(SET CMP0079 NEW)

set(Torch_DIR "${THIRD_PARTY_ROOT}/libtorch")
list(APPEND CMAKE_PREFIX_PATH ${Torch_DIR})
find_package(Torch REQUIRED)


# Build Configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gdwarf-4 -D_GLIBCXX_USE_CXX11_ABI=0")
add_compile_options(-fno-builtin-log)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

# Build Type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif ()


# Include MatrixGraph library
add_subdirectory(${THIRD_PARTY_ROOT}/MatrixGraph)


# Add examples directory
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
	PUBLIC
        matrixgraph_core
	"${TORCH_LIBRARIES}"
	dl
	pthread
	TBB::tbb
	-static-libgcc
	-static-libstdc++
        )
target_include_directories(
	${PROJECT_NAME}
	PRIVATE
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        "${THIRD_PARTY_ROOT}/MatrixGraph"
        "${THIRD_PARTY_ROOT}"
	"${TORCH_INCLUDE_DIRECTORIES}"
	)
