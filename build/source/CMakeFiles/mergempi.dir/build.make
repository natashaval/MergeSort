# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/natashaval/MergeSort

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/natashaval/MergeSort/build

# Include any dependencies generated for this target.
include source/CMakeFiles/mergempi.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include source/CMakeFiles/mergempi.dir/compiler_depend.make

# Include the progress variables for this target.
include source/CMakeFiles/mergempi.dir/progress.make

# Include the compile flags for this target's objects.
include source/CMakeFiles/mergempi.dir/flags.make

source/CMakeFiles/mergempi.dir/clean:
	cd /home/natashaval/MergeSort/build/source && $(CMAKE_COMMAND) -P CMakeFiles/mergempi.dir/cmake_clean.cmake
.PHONY : source/CMakeFiles/mergempi.dir/clean

source/CMakeFiles/mergempi.dir/depend:
	cd /home/natashaval/MergeSort/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/natashaval/MergeSort /home/natashaval/MergeSort/source /home/natashaval/MergeSort/build /home/natashaval/MergeSort/build/source /home/natashaval/MergeSort/build/source/CMakeFiles/mergempi.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : source/CMakeFiles/mergempi.dir/depend

