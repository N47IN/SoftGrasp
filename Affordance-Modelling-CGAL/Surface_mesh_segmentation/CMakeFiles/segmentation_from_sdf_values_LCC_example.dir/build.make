# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/navin/CGAL-5.6.1/examples/Surface_mesh_segmentation

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/navin/CGAL-5.6.1/examples/Surface_mesh_segmentation

# Include any dependencies generated for this target.
include CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/flags.make

CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/segmentation_from_sdf_values_LCC_example.cpp.o: CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/flags.make
CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/segmentation_from_sdf_values_LCC_example.cpp.o: segmentation_from_sdf_values_LCC_example.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/navin/CGAL-5.6.1/examples/Surface_mesh_segmentation/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/segmentation_from_sdf_values_LCC_example.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/segmentation_from_sdf_values_LCC_example.cpp.o -c /home/navin/CGAL-5.6.1/examples/Surface_mesh_segmentation/segmentation_from_sdf_values_LCC_example.cpp

CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/segmentation_from_sdf_values_LCC_example.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/segmentation_from_sdf_values_LCC_example.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/navin/CGAL-5.6.1/examples/Surface_mesh_segmentation/segmentation_from_sdf_values_LCC_example.cpp > CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/segmentation_from_sdf_values_LCC_example.cpp.i

CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/segmentation_from_sdf_values_LCC_example.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/segmentation_from_sdf_values_LCC_example.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/navin/CGAL-5.6.1/examples/Surface_mesh_segmentation/segmentation_from_sdf_values_LCC_example.cpp -o CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/segmentation_from_sdf_values_LCC_example.cpp.s

# Object files for target segmentation_from_sdf_values_LCC_example
segmentation_from_sdf_values_LCC_example_OBJECTS = \
"CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/segmentation_from_sdf_values_LCC_example.cpp.o"

# External object files for target segmentation_from_sdf_values_LCC_example
segmentation_from_sdf_values_LCC_example_EXTERNAL_OBJECTS =

segmentation_from_sdf_values_LCC_example: CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/segmentation_from_sdf_values_LCC_example.cpp.o
segmentation_from_sdf_values_LCC_example: CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/build.make
segmentation_from_sdf_values_LCC_example: /usr/lib/x86_64-linux-gnu/libgmpxx.so
segmentation_from_sdf_values_LCC_example: /usr/lib/x86_64-linux-gnu/libmpfr.so
segmentation_from_sdf_values_LCC_example: /usr/lib/x86_64-linux-gnu/libgmp.so
segmentation_from_sdf_values_LCC_example: CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/navin/CGAL-5.6.1/examples/Surface_mesh_segmentation/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable segmentation_from_sdf_values_LCC_example"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/build: segmentation_from_sdf_values_LCC_example

.PHONY : CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/build

CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/cmake_clean.cmake
.PHONY : CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/clean

CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/depend:
	cd /home/navin/CGAL-5.6.1/examples/Surface_mesh_segmentation && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/navin/CGAL-5.6.1/examples/Surface_mesh_segmentation /home/navin/CGAL-5.6.1/examples/Surface_mesh_segmentation /home/navin/CGAL-5.6.1/examples/Surface_mesh_segmentation /home/navin/CGAL-5.6.1/examples/Surface_mesh_segmentation /home/navin/CGAL-5.6.1/examples/Surface_mesh_segmentation/CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/segmentation_from_sdf_values_LCC_example.dir/depend

