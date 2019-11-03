CXXFLAGS := -g -std=c++17 -MMD

SRCS := $(wildcard *.cc)
DEPS := $(SRCS:.cc=.d)

all : SVD

-include $(DEPS)
