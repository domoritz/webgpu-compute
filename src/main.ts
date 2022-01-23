import "./style.css";

const outputDiv = document.querySelector<HTMLDivElement>("#output")!;

(async () => {
	if (!navigator.gpu) {
		console.log(
			"WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag."
		);
		return;
	}

	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) {
		console.log("Failed to get GPU adapter.");
		return;
	}
	const device = await adapter.requestDevice();

	const input = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);

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
	console.log(new Float32Array(arrayBuffer));

	outputDiv.innerHTML = new Float32Array(arrayBuffer).toString();
})();
