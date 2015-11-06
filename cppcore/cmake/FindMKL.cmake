set(MKLROOT_PATH $ENV{MKLROOT})
if(NOT MKLROOT_PATH)
	set(MKLROOT_PATH "/opt/intel/mkl")
endif()

find_path(MKL_INCLUDE_DIR mkl.h
          PATHS ${MKLROOT_PATH}/include ${MKLROOT_PATH}/mkl/include)
