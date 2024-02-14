# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/hpc_software/tools/cmake/3.26.0/bin/cmake

# The command to remove a file.
RM = /opt/hpc_software/tools/cmake/3.26.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /nfs/site/home/yuankai/code/tiny-dpcpp-nn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /nfs/site/home/yuankai/code/tiny-dpcpp-nn/build_arc

# Include any dependencies generated for this target.
include test/CMakeFiles/test-lib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/test-lib.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/test-lib.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/test-lib.dir/flags.make

test/CMakeFiles/test-lib.dir/main.cpp.o: test/CMakeFiles/test-lib.dir/flags.make
test/CMakeFiles/test-lib.dir/main.cpp.o: /nfs/site/home/yuankai/code/tiny-dpcpp-nn/test/main.cpp
test/CMakeFiles/test-lib.dir/main.cpp.o: test/CMakeFiles/test-lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/nfs/site/home/yuankai/code/tiny-dpcpp-nn/build_arc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/test-lib.dir/main.cpp.o"
	cd /nfs/site/home/yuankai/code/tiny-dpcpp-nn/build_arc/test && /opt/intel/oneapi/2024.0/bin/mpiicpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test-lib.dir/main.cpp.o -MF CMakeFiles/test-lib.dir/main.cpp.o.d -o CMakeFiles/test-lib.dir/main.cpp.o -c /nfs/site/home/yuankai/code/tiny-dpcpp-nn/test/main.cpp

test/CMakeFiles/test-lib.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test-lib.dir/main.cpp.i"
	cd /nfs/site/home/yuankai/code/tiny-dpcpp-nn/build_arc/test && /opt/intel/oneapi/2024.0/bin/mpiicpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /nfs/site/home/yuankai/code/tiny-dpcpp-nn/test/main.cpp > CMakeFiles/test-lib.dir/main.cpp.i

test/CMakeFiles/test-lib.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test-lib.dir/main.cpp.s"
	cd /nfs/site/home/yuankai/code/tiny-dpcpp-nn/build_arc/test && /opt/intel/oneapi/2024.0/bin/mpiicpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /nfs/site/home/yuankai/code/tiny-dpcpp-nn/test/main.cpp -o CMakeFiles/test-lib.dir/main.cpp.s

# Object files for target test-lib
test__lib_OBJECTS = \
"CMakeFiles/test-lib.dir/main.cpp.o"

# External object files for target test-lib
test__lib_EXTERNAL_OBJECTS =

test/libtest-lib.a: test/CMakeFiles/test-lib.dir/main.cpp.o
test/libtest-lib.a: test/CMakeFiles/test-lib.dir/build.make
test/libtest-lib.a: test/CMakeFiles/test-lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/nfs/site/home/yuankai/code/tiny-dpcpp-nn/build_arc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libtest-lib.a"
	cd /nfs/site/home/yuankai/code/tiny-dpcpp-nn/build_arc/test && $(CMAKE_COMMAND) -P CMakeFiles/test-lib.dir/cmake_clean_target.cmake
	cd /nfs/site/home/yuankai/code/tiny-dpcpp-nn/build_arc/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-lib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/test-lib.dir/build: test/libtest-lib.a
.PHONY : test/CMakeFiles/test-lib.dir/build

test/CMakeFiles/test-lib.dir/clean:
	cd /nfs/site/home/yuankai/code/tiny-dpcpp-nn/build_arc/test && $(CMAKE_COMMAND) -P CMakeFiles/test-lib.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/test-lib.dir/clean

test/CMakeFiles/test-lib.dir/depend:
	cd /nfs/site/home/yuankai/code/tiny-dpcpp-nn/build_arc && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /nfs/site/home/yuankai/code/tiny-dpcpp-nn /nfs/site/home/yuankai/code/tiny-dpcpp-nn/test /nfs/site/home/yuankai/code/tiny-dpcpp-nn/build_arc /nfs/site/home/yuankai/code/tiny-dpcpp-nn/build_arc/test /nfs/site/home/yuankai/code/tiny-dpcpp-nn/build_arc/test/CMakeFiles/test-lib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/test-lib.dir/depend

