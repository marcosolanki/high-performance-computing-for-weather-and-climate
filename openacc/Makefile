-include ../config.mk


main: main.cpp utils.hpp
	$(CXX) -o $@ $< $(CXXFLAGS)

.PHONY: clean
clean:
	$(RM) main in_field.csv out_field.csv plot.png
