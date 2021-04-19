CXX := clang++
CPPFLAGS := -isystem /usr/include/opencv4
CXXFLAGS := -O2 -march=native -std=c++17 -MMD -fopenmp -Wall -Wextra
LDFLAGS := -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

SRCS := $(wildcard *.cc)
DEPS := $(SRCS:.cc=.d)

.PHONY: all clean veryclean

all : SVD imProc

% : %.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) -o $@ 

clean :
	@rm -vf SVD

veryclean : clean
	@rm -vf $(DEPS)

-include $(DEPS)
