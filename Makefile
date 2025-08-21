# FANN Neural Network Training Tools Makefile
# Optimized compilation settings for neural network training

CC = gcc
CXX = g++
CFLAGS = -O3 -g -Wall -Wextra
CXXFLAGS = -O3 -g -Wall -Wextra -fpermissive
LIBS = -lfann -lm

# Common object file
COMMON_OBJ = fann_common.o

# Core utilities
CORE_TARGETS = train.exe create.exe run.exe data.exe info.exe

# Optional utilities (uncomment to build)
OPTIONAL_TARGETS = findnet.exe mutate.exe fann_nor.exe cascade.exe

.PHONY: all core optional clean

all: core

core: $(CORE_TARGETS)

optional: $(OPTIONAL_TARGETS)

# Common object file
fann_common.o: fann_common.c fann_common.h
	$(CC) $(CFLAGS) -c $< -o $@

train.exe: train.cpp $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) $< $(COMMON_OBJ) -o $@ $(LIBS)

create.exe: create.c $(COMMON_OBJ)
	$(CC) $(CFLAGS) $< $(COMMON_OBJ) -o $@ $(LIBS)

run.exe: run.c $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) $< $(COMMON_OBJ) -o $@ $(LIBS)

data.exe: data.cpp $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) $< $(COMMON_OBJ) -o $@ $(LIBS)

info.exe: info.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

findnet.exe: find.cpp $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) $< $(COMMON_OBJ) -o $@ $(LIBS)

mutate.exe: mutate.cpp $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) $< $(COMMON_OBJ) -o $@ $(LIBS)

fann_nor.exe: fann_normal.c $(COMMON_OBJ)
	$(CC) $(CFLAGS) $< $(COMMON_OBJ) -o $@ $(LIBS)

cascade.exe: cascade.c $(COMMON_OBJ)
	$(CC) $(CFLAGS) $< $(COMMON_OBJ) -o $@ $(LIBS)

clean:
	rm -f *.exe *.o *.backup

install-deps:
	@echo "Install FANN library:"
	@echo "Ubuntu/Debian: sudo apt-get install libfann-dev"
	@echo "CentOS/RHEL: sudo yum install fann-devel"
	@echo "macOS: brew install fann"
