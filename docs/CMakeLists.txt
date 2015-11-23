add_custom_target(docs COMMAND sphinx-build -b html docs build/docs/html
                  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
