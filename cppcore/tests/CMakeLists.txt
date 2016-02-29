function(download_catch VERSION DIRECTORY)
    set(dest_file "${DIRECTORY}/catch.hpp")

    if(NOT EXISTS ${dest_file})
        message(STATUS "Downloading Catch ${VERSION}...")
        set(url "https://raw.githubusercontent.com/philsquared/Catch/v${VERSION}/single_include/catch.hpp")
        file(DOWNLOAD ${url} ${dest_file} STATUS status)

        if(status)
            execute_process(COMMAND wget -q -O ${dest_file} ${url})
        endif()

        if(NOT EXISTS ${dest_file})
            message(FATAL_ERROR "Could not download Catch ${VERSION}")
        endif()
    endif()
endfunction()

set(CATCH_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../deps/catch/")
download_catch(1.3.4 ${CATCH_INCLUDE_DIR})

add_executable(catch
    catch.cpp
    test_system.cpp
)
target_include_directories(catch PRIVATE ${CATCH_INCLUDE_DIR})
target_link_libraries(catch PRIVATE pybinding_cppcore)

add_custom_target(cpptest COMMAND $<TARGET_FILE:catch> \${ARGS})
