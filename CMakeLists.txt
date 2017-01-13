project(KDTree)

cmake_minimum_required(VERSION 2.8)

# Opencv depends
if(WIN32)
	add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
	find_path(OpenCV_DIR OpenCVConfig.cmake PATHS 
	          $ENV{OpenCV_DIR}
			  "C:/OpenCV/opencv-2.4.9/build" # PC Amaury Mines
			  )
	if( ${OpenCV_DIR} STREQUAL "OpenCV_DIR-NOTFOUND" ) 
		message( FATAL_ERROR "FATAL_ERROR : OpenCV is not found, Please insert your OpenCV Path in CMakeLists.txt")
	else ()
		message(STATUS "Your OpenCV_DIR is ${OpenCV_DIR}")
	endif()
	find_package(OpenCV REQUIRED)
else()
	find_package(OpenCV REQUIRED)
endif()
include_directories( ${OPENCV_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )

add_executable(test2D src/main_test.cpp )
target_link_libraries(test2D ${OpenCV_LIBS} )

