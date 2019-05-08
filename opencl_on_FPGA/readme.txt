1.files:
host-  run on the CPU platform
device - run on the FPGA

2.host
main.cpp      - the main contents to control the program flow
layer_config.h- parameters configraton

3.device
conv_pipe.cl - the main contents executed on the FPGA
hw_param.cl  -  hardware parameters
/RTL/rtl_lib.h - two module written in verilog language
                 the first is  multiplier and adder circuit
                 the second is gauss random generator
