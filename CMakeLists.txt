cmake_minimum_required(VERSION 3.16)
project(MLPCpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Options ────────────────────────────────────────────────────────
option(MLPCPP_AD_TEST    "Build AD smoke tests"             ON)
option(MLPCPP_CODI_NN    "Build CoDi CNeuralNetwork test"   ON)

# ── Locate SU2 and MLPCpp roots ────────────────────────────────────
get_filename_component(SU2_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    ABSOLUTE)

get_filename_component(MLPCPP_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ABSOLUTE)

message(STATUS "SU2 root   : ${SU2_ROOT}")
message(STATUS "MLPCpp root: ${MLPCPP_ROOT}")

# ── Common includes ────────────────────────────────────────────────
set(COMMON_INC
    ${MLPCPP_ROOT}/include
    ${SU2_ROOT}/Common/include
    ${SU2_ROOT}/externals/catch2
    ${SU2_ROOT}/externals/codi/include
    ${SU2_ROOT}/externals/medi/include
)

# ── Common compile definitions ─────────────────────────────────────
set(COMMON_DEFS
    CODI_REVERSE_TYPE
    HAVE_CODI
)

# ── SU2 AD core ────────────────────────────────────────────────────
set(SU2_AD_LIB ${SU2_ROOT}/build/SU2_CFD/src/libSU2core_AD.a)

# ── Helper macro ───────────────────────────────────────────────────
macro(add_ad_test TARGET SOURCE)
    add_executable(${TARGET} UnitTests/${SOURCE})
    target_include_directories(${TARGET} PRIVATE ${COMMON_INC})
    target_compile_definitions(${TARGET} PRIVATE ${COMMON_DEFS})
    target_compile_options(${TARGET}     PRIVATE -O2 -Wall -Wextra)
    target_link_libraries(${TARGET}      PRIVATE ${SU2_AD_LIB})
    add_test(NAME ${TARGET} COMMAND ${TARGET})
endmacro()

enable_testing()

# ── Existing tests ─────────────────────────────────────────────────
if(MLPCPP_AD_TEST)
    
endif()

# ── New: CoDi on loaded CNeuralNetwork ─────────────────────────────
if(MLPCPP_CODI_NN)

    add_executable(mlpcpp_codi_nn_test
        UnitTests/test_codi_neuralnetwork.cpp
    )

    target_include_directories(mlpcpp_codi_nn_test PRIVATE
        ${COMMON_INC}
    )

    target_compile_definitions(mlpcpp_codi_nn_test PRIVATE
        ${COMMON_DEFS}
    )

    target_compile_options(mlpcpp_codi_nn_test PRIVATE
        -O2 -Wall -Wextra
    )

    # Links against SU2 AD core for the AD::* wrappers
    # AND the CoDi headers-only library (already in COMMON_INC)
    target_link_libraries(mlpcpp_codi_nn_test PRIVATE
        ${SU2_AD_LIB}
    )

    # ctest: pass the MLP file as a command-line argument
    # Adjust the path to wherever MLP_test.mlp lives relative to the build dir
    add_test(
        NAME    mlpcpp_codi_nn
        COMMAND mlpcpp_codi_nn_test
                ${MLPCPP_ROOT}/MLP_test.mlp
        WORKING_DIRECTORY ${MLPCPP_ROOT}
    )

endif()