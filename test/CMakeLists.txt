link_libraries(Catch2::Catch2 lzlearn)
foreach(NAME thread_pool data_set sampler optimizer model)
	add_executable(test_${NAME} catch2.cpp test_${NAME}.cpp)
	catch_discover_tests(test_${NAME})
endforeach()
#add_executable(tests
#		catch2.cpp
#		test_thread_pool.cpp
#		test_data_set.cpp
#		test_sampler.cpp
#		test_optimizer.cpp
#		test_model.cpp)
