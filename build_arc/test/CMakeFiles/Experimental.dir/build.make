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

# Utility rule file for Experimental.

# Include any custom commands dependencies for this target.
include test/CMakeFiles/Experimental.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/Experimental.dir/progress.make

test/CMakeFiles/Experimental:
	cd /nfs/site/home/yuankai/code/tiny-dpcpp-nn/build_arc/test && /opt/hpc_software/tools/cmake/3.26.0/bin/ctest -D Experimental

Experimental: test/CMakeFiles/Experimental
Experimental: test/CMakeFiles/Experimental.dir/build.make
.PHONY : Experimental

# Rule to build all files generated by this target.
test/CMakeFiles/Experimental.dir/build: Experimental
.PHONY : test/CMakeFiles/Experimental.dir/build

test/CMakeFiles/Experimental.dir/clean:
	cd /nfs/site/home/yuankai/code/tiny-dpcpp-nn/build_arc/test && $(CMAKE_COMMAND) -P CMakeFiles/Experimental.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/Experimental.dir/clean

test/CMakeFiles/Experimental.dir/depend:
	cd /nfs/site/home/yuankai/code/tiny-dpcpp-nn/build_arc && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /nfs/site/home/yuankai/code/tiny-dpcpp-nn /nfs/site/home/yuankai/code/tiny-dpcpp-nn/test /nfs/site/home/yuankai/code/tiny-dpcpp-nn/build_arc /nfs/site/home/yuankai/code/tiny-dpcpp-nn/build_arc/test /nfs/site/home/yuankai/code/tiny-dpcpp-nn/build_arc/test/CMakeFiles/Experimental.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/Experimental.dir/depend

