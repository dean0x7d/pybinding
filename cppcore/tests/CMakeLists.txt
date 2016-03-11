add_executable(catch
    catch.cpp
    test_system.cpp
    test_support.cpp
)
target_include_directories(catch PRIVATE ${CATCH_INCLUDE_DIR})
target_link_libraries(catch PRIVATE pybinding_cppcore)

add_custom_target(cpptest COMMAND $<TARGET_FILE:catch> \${ARGS})
