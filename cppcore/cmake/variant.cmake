set(VARIANT_VERSION 1.1.0)
set(VARIANT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/variant")

if(NOT EXISTS ${VARIANT_INCLUDE_DIR})
    message(STATUS "Downloading mapbox variant v${VARIANT_VERSION}...")
    foreach(filename variant.hpp recursive_wrapper.hpp)
        set(repo "https://raw.githubusercontent.com/mapbox/variant")
        download("${repo}/v${VARIANT_VERSION}/${filename}"
                 "${VARIANT_INCLUDE_DIR}/${filename}")
    endforeach()
endif()
