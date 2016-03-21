add_custom_target(docs COMMAND sphinx-build -b html docs ${CMAKE_CURRENT_BINARY_DIR}/html
                  DEPENDS _pybinding WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
