-include ../config.mk


main: main.cu wrappers.cuh kernels.cuh utils.cuh
	$(CUDACXX) -o $@ $< $(CUDAFLAGS)

.PHONY: clean
clean:
	$(RM) main in_field.csv out_field.csv
