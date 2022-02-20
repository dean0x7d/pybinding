download_dependency(fmt 3.0.2 https://raw.githubusercontent.com/fmtlib/fmt/\${VERSION}
                    fmt/format.h fmt/format.cc fmt/ostream.h fmt/ostream.cc)

add_library(fmt STATIC EXCLUDE_FROM_ALL
            ${FMT_INCLUDE_DIR}/fmt/format.cc
            ${FMT_INCLUDE_DIR}/fmt/ostream.cc)
target_include_directories(fmt SYSTEM PUBLIC ${FMT_INCLUDE_DIR})
set_target_properties(fmt PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
