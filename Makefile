SRC = $(wildcard source/*.cpp)
OBJ = $(SRC:.cpp=.o)
GPU = pvc

program: $(OBJ)
        icpx -fsycl -qmkl=parallel -I include/ -I include/Network -I include/Losses -I include/Optimizers -fsycl-targets=spir64_gen -Xs "-device $(GPU)" $^ -o $@

%.o: %.cpp
        icpx -fsycl -qmkl=parallel -I include/ -I include/Network -I include/Losses -I include/Optimizers -fsycl-targets=spir64_gen -c $< -o $@

clean:
        rm -fr *.o

pvc_build:
        tail -n +8 source/SwiftNetMLP.cpp > temp && mv temp source/SwiftNetMLP.cpp
        cat pvc_header | cat - SwiftNetMLP.cpp > temp && mv temp source/SwiftNetMLP.cpp

dg2_build:
        tail -n +8 source/SwiftNetMLP.cpp > temp && mv temp source/SwiftNetMLP.cpp
        cat dg2_header | cat - source/SwiftNetMLP.cpp > temp && mv temp source/SwiftNetMLP.cpp
        $(eval GPU = dg2-g12)
        @echo "$(GPU)"

dg2: dg2_build program

pvc: pvc_build program
.PHONY: clean dg2 pvc dg2_build pvc_build
