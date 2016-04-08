download_dependency(cppformat 2.0.0
                    https://raw.githubusercontent.com/cppformat/cppformat/\${VERSION}
                    format.h format.cc)

add_library(cppformat STATIC EXCLUDE_FROM_ALL ${CPPFORMAT_INCLUDE_DIR}/format.cc)
target_include_directories(cppformat SYSTEM PUBLIC ${CPPFORMAT_INCLUDE_DIR})
if(NOT WIN32)
    target_compile_options(cppformat PUBLIC -std=c++11)
endif()
set_target_properties(cppformat PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
