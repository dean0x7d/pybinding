set(EIGEN3_VERSION 3.2.8)
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/eigen3")

if(NOT EXISTS ${EIGEN3_INCLUDE_DIR})
    message(STATUS "Downloading eigen v${EIGEN3_VERSION}...")
    set(tmp_dir "${CMAKE_CURRENT_SOURCE_DIR}/deps/tmp")
    set(tar_file "${tmp_dir}/eigen.tar.gz")
    download("https://bitbucket.org/eigen/eigen/get/${EIGEN3_VERSION}.tar.gz" ${tar_file})

    # extract downloaded tar file
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${tar_file} WORKING_DIRECTORY ${tmp_dir})
    file(REMOVE ${tar_file})
    file(GLOB ex_dir ${tmp_dir}/eigen-*)

    if(NOT EXISTS ${ex_dir})
        message(FATAL_ERROR "Error downloading eigen v${EIGEN3_VERSION}")
    else()
        # rename to a nicer dir
        file(RENAME ${ex_dir} ${EIGEN3_INCLUDE_DIR})
        # delete everything except the headers and signature file
        file(GLOB to_delete ${EIGEN3_INCLUDE_DIR}/*)
        list(REMOVE_ITEM to_delete ${EIGEN3_INCLUDE_DIR}/Eigen)
        list(REMOVE_ITEM to_delete ${EIGEN3_INCLUDE_DIR}/signature_of_eigen3_matrix_library)
        file(REMOVE_RECURSE ${to_delete})
        file(REMOVE_RECURSE ${tmp_dir})
    endif()
endif()
