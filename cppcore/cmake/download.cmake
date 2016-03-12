# Conda's cmake does not support https so the regular
# file DOWNLOAD may fail. This adds a fallback path.
function(download URL DESTINATION)
    file(DOWNLOAD ${URL} ${DESTINATION} STATUS status)
    if(status)
        execute_process(COMMAND wget -q -O ${DESTINATION} ${URL}
                        RESULT_VARIABLE status)
    endif()
    if(status)
        execute_process(COMMAND curl --create-dirs -s -S -o ${DESTINATION} ${URL}
                        RESULT_VARIABLE status)
    endif()
    if(status)
        message(FATAL_ERROR "Could not download ${URL}")
    endif()
endfunction()
