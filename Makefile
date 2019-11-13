CXXFLAGS := -g -std=c++17 -MMD

SRCS := $(wildcard *.cc)
DEPS := $(SRCS:.cc=.d)

.PHONY: all clean veryclean

all : SVD

clean :
	@rm -vf SVD

veryclean : clean
	@rm -vf $(DEPS)

-include $(DEPS)
