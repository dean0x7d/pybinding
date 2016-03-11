set(CPPFORMAT_VERSION 2.0.0)
set(CPPFORMAT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/cppformat")

if(NOT EXISTS ${CPPFORMAT_INCLUDE_DIR})
    message(STATUS "Downloading cppformat v${CPPFORMAT_VERSION}...")
    foreach(filename format.h format.cc)
        set(repo "https://raw.githubusercontent.com/cppformat/cppformat")
        download("${repo}/${CPPFORMAT_VERSION}/${filename}"
                 "${CPPFORMAT_INCLUDE_DIR}/${filename}")
    endforeach()
endif()

add_library(cppformat STATIC EXCLUDE_FROM_ALL ${CPPFORMAT_INCLUDE_DIR}/format.cc)
target_include_directories(cppformat SYSTEM PUBLIC ${CPPFORMAT_INCLUDE_DIR})
if(NOT WIN32)
    target_compile_options(cppformat PUBLIC -std=c++11)
endif()
set_target_properties(cppformat PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
