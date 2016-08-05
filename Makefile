CC = g++
PWD = $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
LIB = -L ${PWD}
OPENCV_LIB = `pkg-config --cflags --libs opencv`

main: stereomatch.cpp
	${CC} -std=c++11 ${LIB} -g -o stereomatch stereomatch.cpp ${OPENCV_LIB}

#main: stereocalibrate.cpp
#	${CC} -std=c++11 ${LIB} -g -o stereocalibrate stereocalibrate.cpp ${OPENCV_LIB}

clean:
	$(RM) *.o *.so main
