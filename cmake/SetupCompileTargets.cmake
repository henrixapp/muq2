# PURPOSE:
# This file sets up the MUQ build targets (e.g., libMuqModeling).  Information is
# used from the compile groups that were processed in the ProcessCompileGroups.cmake
# file.
#

set(MUQ_LIBRARIES )
set(MUQ_PYTHON_LIBRARIES )

# Build all the targets
foreach(libName ${MUQ_TARGETS})

    list(LENGTH ${libName}_SOURCES strLength)
    if(${strLength} GREATER 0)

        string(REGEX MATCH "^pymuq" IsPythonWrapper ${libName})

        if(IsPythonWrapper)
            pybind11_add_module(${libName} SHARED NO_EXTRAS ${${libName}_SOURCES})
            list(APPEND MUQ_PYTHON_LIBRARIES ${libName})

            #string(SUBSTRING ${libname} 2 -1 CppLib)
            #target_link_libraries(${CppLib} ${libname})

        else()
            ADD_LIBRARY(${libName} ${${libName}_SOURCES})
            list(APPEND MUQ_LIBRARIES ${libName})
        endif()
        
        TARGET_LINK_LIBRARIES(${libName} PUBLIC ${MUQ_LINK_LIBS})

        install(TARGETS ${libName}
                EXPORT ${CMAKE_PROJECT_NAME}Depends
                LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
                ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
    endif()
    
endforeach()

INSTALL (
    DIRECTORY ${CMAKE_SOURCE_DIR}/MUQ
    DESTINATION include
    FILES_MATCHING PATTERN "*.h")
    
# If a group depends on an external library that is going to be built by MUQ, then make sure we account for that dependency
foreach(group ${MUQ_GROUPS})

    if(${group}_IS_COMPILED)
        list(LENGTH ${group}_SOURCES strLength)

        foreach(depend ${POSSIBLE_MUQ_DEPENDENCIES})
            list(FIND ${group}_REQUIRES ${depend} needsExternal)

            if(USE_INTERNAL_${depend})
                if(needsExternal AND ${USE_INTERNAL_${depend}} AND (strLength GREATER 0))
                    add_dependencies(${${group}_LIBRARY} ${depend})
                endif()
	    endif()
        endforeach()

        # Add dependencies between different MUQ libraries
        foreach(depend ${${group}_REQUIRES_GROUPS})
        
            if(NOT ${${group}_LIBRARY} STREQUAL ${${depend}_LIBRARY})
                IF( ${depend}_IS_COMPILED )
                    message(STATUS "Trying to add connection between ${${group}_LIBRARY} and ${${depend}_LIBRARY}")
                    target_link_libraries(${${group}_LIBRARY} PUBLIC ${${depend}_LIBRARY})
                    add_dependencies(${${group}_LIBRARY} ${${depend}_LIBRARY})
                endif()
            endif()
            
        endforeach()
    endif()
    
endforeach()
