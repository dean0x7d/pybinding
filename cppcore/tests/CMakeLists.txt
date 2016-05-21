add_executable(catch
    catch.cpp
    fixtures.hpp
    fixtures.cpp
    test_compute.cpp
    test_detail.cpp
    test_greens.cpp
    test_leads.cpp
    test_numeric.cpp
    test_shape.cpp
    test_system.cpp
)
target_include_directories(catch PRIVATE ${CATCH_INCLUDE_DIR})
target_link_libraries(catch PRIVATE pybinding_cppcore)

add_custom_target(cpptest COMMAND $<TARGET_FILE:catch> \${ARGS})
