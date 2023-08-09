SRC = $(wildcard *.cpp)
OBJ = $(SRC:.cpp=.o)


program: $(OBJ)
	icpx -fsycl -qmkl=parallel -fsycl-targets=spir64_gen -Xs "-device dg2-g12" $^ -o $@

%.o: %.cpp
	icpx -fsycl -qmkl=parallel -fsycl-targets=spir64_gen -c $< -o $@

clean :
	rm -fr *.o

.PHONY: clean
