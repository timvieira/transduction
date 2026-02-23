cmake_minimum_required(VERSION 3.15)
project(openfst_decomp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Vendored OpenFST (static library, no scripts/bins, minimal footprint)
# ---------------------------------------------------------------------------

set(OPENFST_SRC ${CMAKE_SOURCE_DIR}/third_party/openfst/src)
set(OPENFST_INCLUDE ${OPENFST_SRC}/include)

# Core OpenFST source files needed for VectorFst / ConstFst / compose / etc.
# We compile only the lib/ directory (no scripts, bins, extensions).
file(GLOB OPENFST_LIB_SRCS ${OPENFST_SRC}/lib/*.cc)

add_library(fst_static STATIC ${OPENFST_LIB_SRCS})
target_include_directories(fst_static PUBLIC ${OPENFST_INCLUDE})
target_compile_definitions(fst_static PRIVATE
    # Disable dynamic registration (we don't need it)
    FST_NO_DYNAMIC_LINKING
)
target_compile_options(fst_static PRIVATE -O2 -fPIC)

# ---------------------------------------------------------------------------
# Our C++ library (links against fst_static)
# ---------------------------------------------------------------------------

set(DECOMP_SRCS
    src/fst_convert.cc
    src/precover_fst.cc
    src/universality.cc
    src/decompose.cc
    src/peekaboo_fst.cc
    src/lazy_dfa.cc
    src/dirty_peekaboo.cc
)

add_library(openfst_decomp STATIC ${DECOMP_SRCS})
target_include_directories(openfst_decomp PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${OPENFST_INCLUDE}
)
target_link_libraries(openfst_decomp PUBLIC fst_static)
target_compile_options(openfst_decomp PRIVATE -O2 -fPIC -Wall -Wextra -Wno-unused-parameter)
