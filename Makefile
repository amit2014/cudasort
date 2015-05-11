CC  = nvcc -arch=compute_35 -code=sm_35 -rdc=true
CXX = $(CC)

CXXFLAGS = -O3 -Xcompiler " -Wall -fopenmp"
LIBFLAGS = -Xcompiler " -fPIC"

LFLAGS = -L.
LIBS = -lpsort

SRCS = check.cu time.cu
LIBSRCS = qsort.cu msort.cu sort.cu common.cu

OBJS = $(SRCS:.cu=.o)
LIBOBJS = $(LIBSRCS:.cu=.o)

LIBRARY = libpsort.so
TEST = check time
.PHONY: clean debug depend

$(LIBRARY): $(LIBOBJS)
	$(CXX) $(CXXFLAGS) -shared -o $(LIBRARY) $(LIBOBJS)

all: $(LIBRARY) $(TEST)
debug: CXXFLAGS += -g -DDEBUG
debug: all
profiler: CXXFLAGS += -Dgprofiler
profiler: LFLAGS += -L/usr/local/lib
profiler: LIBS += -lprofiler
profiler: debug

$(TEST): %: %.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LFLAGS) $(LIBS)

$(LIBOBJS): %.o: %.cu
	$(CXX) $(CXXFLAGS) $(LIBFLAGS) -c $< -o $@

clean:
	$(RM) *.o *~ $(TEST) $(LIBRARY)

depend: $(SRCS) $(LIBSRCS)
	makedepend -Y. $^

# DO NOT DELETE THIS LINE -- make depend needs it

check.o: common.h sort.h payloadsize.h
time.o: common.h sort.h payloadsize.h
qsort.o: sort.h payloadsize.h
msort.o: sort.h payloadsize.h common.h
sort.o: sort.h payloadsize.h common.h
common.o: common.h sort.h payloadsize.h
