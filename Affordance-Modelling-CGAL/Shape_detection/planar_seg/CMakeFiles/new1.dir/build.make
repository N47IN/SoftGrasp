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
CMAKE_SOURCE_DIR = /home/navin/CGAL-5.6.1/examples/Shape_detection/planar_seg

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/navin/CGAL-5.6.1/examples/Shape_detection/planar_seg

# Include any dependencies generated for this target.
include CMakeFiles/new1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/new1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/new1.dir/flags.make

CMakeFiles/new1.dir/region_growing_planes_on_point_set_3.cpp.o: CMakeFiles/new1.dir/flags.make
CMakeFiles/new1.dir/region_growing_planes_on_point_set_3.cpp.o: region_growing_planes_on_point_set_3.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/navin/CGAL-5.6.1/examples/Shape_detection/planar_seg/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/new1.dir/region_growing_planes_on_point_set_3.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/new1.dir/region_growing_planes_on_point_set_3.cpp.o -c /home/navin/CGAL-5.6.1/examples/Shape_detection/planar_seg/region_growing_planes_on_point_set_3.cpp

CMakeFiles/new1.dir/region_growing_planes_on_point_set_3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/new1.dir/region_growing_planes_on_point_set_3.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/navin/CGAL-5.6.1/examples/Shape_detection/planar_seg/region_growing_planes_on_point_set_3.cpp > CMakeFiles/new1.dir/region_growing_planes_on_point_set_3.cpp.i

CMakeFiles/new1.dir/region_growing_planes_on_point_set_3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/new1.dir/region_growing_planes_on_point_set_3.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/navin/CGAL-5.6.1/examples/Shape_detection/planar_seg/region_growing_planes_on_point_set_3.cpp -o CMakeFiles/new1.dir/region_growing_planes_on_point_set_3.cpp.s

# Object files for target new1
new1_OBJECTS = \
"CMakeFiles/new1.dir/region_growing_planes_on_point_set_3.cpp.o"

# External object files for target new1
new1_EXTERNAL_OBJECTS =

new1: CMakeFiles/new1.dir/region_growing_planes_on_point_set_3.cpp.o
new1: CMakeFiles/new1.dir/build.make
new1: /usr/lib/x86_64-linux-gnu/libgmpxx.so
new1: /usr/lib/x86_64-linux-gnu/libmpfr.so
new1: /usr/lib/x86_64-linux-gnu/libgmp.so
new1: CMakeFiles/new1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/navin/CGAL-5.6.1/examples/Shape_detection/planar_seg/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable new1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/new1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/new1.dir/build: new1

.PHONY : CMakeFiles/new1.dir/build

CMakeFiles/new1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/new1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/new1.dir/clean

CMakeFiles/new1.dir/depend:
	cd /home/navin/CGAL-5.6.1/examples/Shape_detection/planar_seg && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/navin/CGAL-5.6.1/examples/Shape_detection/planar_seg /home/navin/CGAL-5.6.1/examples/Shape_detection/planar_seg /home/navin/CGAL-5.6.1/examples/Shape_detection/planar_seg /home/navin/CGAL-5.6.1/examples/Shape_detection/planar_seg /home/navin/CGAL-5.6.1/examples/Shape_detection/planar_seg/CMakeFiles/new1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/new1.dir/depend

