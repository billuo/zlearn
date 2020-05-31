link_libraries(external)

add_subdirectory(base)
link_libraries(base)

add_library(lzlearn INTERFACE) # one-in-all lib for building application
target_include_directories(lzlearn INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

foreach (LIB data model optimizer)
	add_subdirectory(${LIB})
	target_include_directories(${LIB} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
	target_link_libraries(lzlearn INTERFACE ${LIB})
endforeach ()

add_executable(zlearn
		main.cpp
		application.cpp
		application.h
		application_impl.h
		train.cpp
		predict.cpp
		cli.h application_impl.cpp)
target_link_libraries(zlearn PRIVATE lzlearn)

add_subdirectory(utility)


