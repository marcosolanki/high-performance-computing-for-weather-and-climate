-include config.mk


main: main.cpp
	$(CXX) -o main main.cpp $(CXXFLAGS)

.PHONY: clean
clean:
	$(RM) main in_field.csv out_field.csv
