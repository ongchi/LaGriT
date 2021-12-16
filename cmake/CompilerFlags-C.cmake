# ======= C flags =========================================
if("${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
  MESSAGE(STATUS "  C compiler: Clang")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w -m64")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
  MESSAGE(STATUS "  C compiler: GNU GCC")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w -m64")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
  MESSAGE(STATUS "  C compiler: Intel C")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
  MESSAGE(STATUS "  C compiler: Microsoft Visual C")
endif()

# ======= C++ flags =======================================
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  message(STATUS "  C++ compiler: Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -m64")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  message(STATUS "  C++ compiler: GNU G++")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -m64")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  message(STATUS "  C++ compiler: Intel C++")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  message(STATUS "  C++ compiler: Microsoft Visual C++")
endif()