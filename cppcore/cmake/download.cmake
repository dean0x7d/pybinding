# Conda's cmake does not support https so the regular
# file DOWNLOAD may fail. This adds a fallback path.
function(download URL DESTINATION)
    file(DOWNLOAD ${URL} ${DESTINATION} STATUS status)
    if(status)
        execute_process(COMMAND wget -q -O ${DESTINATION} ${URL})
    endif()
    if(NOT EXISTS ${DESTINATION})
        message(FATAL_ERROR "Could not download ${URL}")
    endif()
endfunction()
