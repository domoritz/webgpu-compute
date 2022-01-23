import "./style.css";

const outputDiv = document.querySelector<HTMLDivElement>("#output")!;

(async () => {
	if (!navigator.gpu) {
		outputDiv.innerHTML =
			"WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.";
		return;
	}

	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) {
		outputDiv.innerHTML = "Failed to get GPU adapter.";
		return;
	}
	const device = await adapter.requestDevice();

	// Compute shader code
	const shaderModule = device.createShaderModule({
		code: `
        struct Array {
          data: array<f32>;
        };

        [[group(0), binding(0)]] var<storage, read> input : Array;
        [[group(0), binding(1)]] var<storage, write> result : Array;

        [[stage(compute), workgroup_size(8)]]
        fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
          result.data[global_id.x] = input.data[global_id.x] * 2.0;
        }
      `,
	});

	async function compute(device: GPUDevice, input: Float32Array) {
		const gpuBufferInput = device.createBuffer({
			mappedAtCreation: true,
			size: input.byteLength,
			usage: GPUBufferUsage.STORAGE,
		});
		const arrayBufferInput = gpuBufferInput.getMappedRange();

		new Float32Array(arrayBufferInput).set(input);
		gpuBufferInput.unmap();

		// Result

		const resultBufferSize = Float32Array.BYTES_PER_ELEMENT * input.length;
		const resultBuffer = device.createBuffer({
			size: resultBufferSize,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		// Pipeline setup

		const computePipeline = device.createComputePipeline({
			compute: {
				module: shaderModule,
				entryPoint: "main",
			},
		});

		// Bind group

		const bindGroup = device.createBindGroup({
			layout: computePipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: {
						buffer: gpuBufferInput,
					},
				},
				{
					binding: 1,
					resource: {
						buffer: resultBuffer,
					},
				},
			],
		});

		// Commands submission

		const commandEncoder = device.createCommandEncoder();

		const passEncoder = commandEncoder.beginComputePass();
		passEncoder.setPipeline(computePipeline);
		passEncoder.setBindGroup(0, bindGroup);

		console.timeStamp("Starting dispatch");
		passEncoder.dispatch(input.length);
		passEncoder.endPass();

		// Get a GPU buffer for reading in an unmapped state.
		const gpuReadBuffer = device.createBuffer({
			size: resultBufferSize,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		// Encode commands for copying buffer to buffer.
		commandEncoder.copyBufferToBuffer(
			resultBuffer,
			0,
			gpuReadBuffer,
			0,
			resultBufferSize
		);

		// Submit GPU commands.
		const gpuCommands = commandEncoder.finish();
		device.queue.submit([gpuCommands]);

		// Read buffer.
		await gpuReadBuffer.mapAsync(GPUMapMode.READ);
		const arrayBuffer = gpuReadBuffer.getMappedRange();

		return new Float32Array(arrayBuffer);
	}

	const input = new Float32Array(10000).map((_, i) => i);

	console.time("GPU");
	const outputGPU = await compute(device, input);
	console.timeEnd("GPU");

	console.time("CPU");
	const outputCPU = new Float32Array(input.length);
	for (let i = 0; i < input.length; i++) {
		outputCPU[i] = i * 2;
	}
	console.timeEnd("CPU");

	console.log(outputGPU);
	console.log(outputCPU);

	outputDiv.innerHTML = "Done! Check the console.";
})();
