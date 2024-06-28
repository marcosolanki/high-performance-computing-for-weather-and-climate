```
  _   _    ____      ____ _  _                   ____       ____      ____    U  ___ u    _    U _____ u   ____   _____   
 |'| |'| U|  _"\ uU /"___| ||"|   __        __U /"___|    U|  _"\ uU |  _"\ u  \/"_ \/ U |"| u \| ___"|/U /"___| |_ " _|  
/| |_| |\\| |_) |/\| | u | || |_  \"\      /"/\| | u      \| |_) |/ \| |_) |/  | | | |_ \| |/   |  _|"  \| | u     | |    
U|  _  |u |  __/   | |/__|__   _| /\ \ /\ / /\ | |/__      |  __/    |  _ <.-,_| |_| | |_| |_,-.| |___   | |/__   /| |\   
 |_| |_|  |_|       \____| /|_|\ U  \ V  V /  U \____|     |_|       |_| \_\\_)-\___/ \___/-(_/ |_____|   \____| u |_|U   
 //   \\  ||>>_    _// \\ u_|||_u.-,_\ /\ /_,-._// \\      ||>>_     //   \\_    \\    _//      <<   >>  _// \\  _// \\_  
(_") ("_)(__)__)  (__)(__)(__)__) \_)-'  '-(_/(__)(__)    (__)__)   (__)  (__)  (__)  (__)     (__) (__)(__)(__)(__) (__) 
```
# High-level, mid-level and low-level GPU programming comparison

## Build Instruction
Just some notes about compiling OpenACC on PizDaint so that I don't forget it.
##### Compile CPP & OpenACC
```
module load daint-gpu
module swap PrgEnv-cray PrgEnv-nvidia
module load craype-accel-nvidia60
```

##### Run It
```
groups #  get account infos
srun --account class03 -n 1 -C GPU ./main 128 128 64 1024
```
