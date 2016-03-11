set(CATCH_VERSION 1.3.5)
set(CATCH_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/catch")

if(NOT EXISTS ${CATCH_INCLUDE_DIR})
    message(STATUS "Downloading catch v${CATCH_VERSION}...")
    set(repo "https://raw.githubusercontent.com/philsquared/Catch")
    download("${repo}/v${CATCH_VERSION}/single_include/catch.hpp"
             "${CATCH_INCLUDE_DIR}/catch.hpp")
endif()
