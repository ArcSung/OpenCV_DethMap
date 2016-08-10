CC = g++
PWD = $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
LIB = -L ${PWD}
LDLIBS = `pkg-config --cflags --libs opencv`

main: stereomatch.cpp
	${CC} -std=c++11 ${LIB} -g -o stereomatch stereomatch.cpp ${LDLIBS}

#main: stereocalibrate.cpp
#	${CC} -std=c++11 ${LIB} -g -o stereocalibrate stereocalibrate.cpp ${OPENCV_LIB}

clean:
	$(RM) *.o *.so main
