# Conda's cmake does not support https so the regular
# file DOWNLOAD may fail. This adds a fallback path.
function(download URL DESTINATION)
    file(DOWNLOAD ${URL} ${DESTINATION} STATUS status)
    list(GET status 0 error)
    if(error)
        execute_process(COMMAND wget -q -O ${DESTINATION} ${URL}
                        RESULT_VARIABLE error)
    endif()
    if(error)
        execute_process(COMMAND curl --create-dirs -s -S -o ${DESTINATION} ${URL}
                        RESULT_VARIABLE error)
    endif()
    if(error)
        message(FATAL_ERROR "Could not download ${URL}")
    endif()
endfunction()
