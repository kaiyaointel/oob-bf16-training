## OOB PT BF16  
Author: Kai Yao kai.yao@intel.com

### Benchmarking
If running on local machines (not through jenkins), simply do:
```
source ./simple_benchmark.sh
```
and result logs will be stored in ./zzzlogs
### P.S.
In each workload folder, don't read README.md for instructions on benchmarking each workload, please refer to each launch_benchmark.sh for instructions.

### Analysis

1. Profile Parser analysis  
   1.1. Add these lines in workload training script:  
	```
	...
	...
	with torch.profiler.profile(
		activities = [torch.profiler.ProfilerActivity.CPU]
	) as prof:
		### autocast lines
		# with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
		#	 loss = model(next(train_loader), return_loss = True)
		#	 loss.backward()
		
	import os
	import pathlib
	timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
	if not os.path.exists(timeline_dir):
		os.makedirs(timeline_dir)
	timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
				str([index_you_use]) + '-' + str(os.getpid()) + '.json'

	prof.export_chrome_trace(timeline_file)
	...
	...
	```
   1.2. Copy ```profile_parser.py``` to the folder where abovementioned .json file is stored, and run ```python profile_parser.py -f xxxxxx.json | tee xxx.log``` to fetch parsing result  
   1.3. Save it to Excel and split it by spaces
2. Benchdnn perf evaluation analysis  
   2.1. Set env var  
	```
	export DNNL_VERBOSE=1
	export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
	```
   2.2. Build onednn (on SPR)  
	```
	git clone https://github.com/oneapi-src/oneDNN && cd oneDNN && mkdir build && cd build
	cmake .. && make -j
	cd tests/benchdnn
	```
   2.3. Run ./benchdnn benchmark tool according to each log line  
	```
	Example log line: dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_bf16::blocked:cdeba:f0 dst_bf16::blocked:abcde:f0,,,128x384x3x3x3,1.12695
	Example bdnn command: ./benchdnn --mode=P --reorder --sdt=bf16 --ddt=bf16 --stag=cdeba --dtag=abcde 128x384x3x3x3
	```
	Baseline: 
	```
	numactl --cpunodebind=0 --membind=0 ./benchdnn --mode=P --ip --cfg=bf16bf16bf16 --batch=/home2/kyao/oneDNN/build/tests/benchdnn/inputs/ip/shapes_bert_large
	numactl --cpunodebind=0 --membind=0 ./benchdnn --mode=P --matmul --cfg=bf16bf16bf16 --batch=/home2/kyao/oneDNN/build/tests/benchdnn/inputs/matmul/shapes_bert_large
	```
	matmul command:
	```
	numactl --cpunodebind=0 --membind=0 ./benchdnn --mode=P --matmul --cfg=bf16bf16bf16 --attr-post-ops=sum:0 --stag=abc --wtag=abc --dtag=abc 384x128x64:384x64x128:384x128x128
	```
	inner-product command:
	```
	numactl --cpunodebind=0 --membind=0 ./benchdnn --mode=P --ip --cfg=bf16bf16bf16 --stag=ab --wtag=AB16b64a2b --dtag=ab mb4096ic128oc768
	```
    More about ./benchdnn command please refer to https://github.com/oneapi-src/oneDNN/blob/master/tests/benchdnn/README.md