import * as ml from "./mathlib/math.ts";
import { Network } from "./neural_network/nnetwork";
import { MnistData } from "./mnist";

export type Batch = {
	xs: ml.Vector;
	ys: ml.Vector;
}

export class Engine {
	network: Network;
	data: MnistData;
	rng: ml.PCG32;

	private readonly img_width = 28;

	private readonly input_size = this.img_width * this.img_width;
	private readonly output_size = 10;

	private readonly num_epochs = 30; // 30
	private readonly batch_size = 80;
	private readonly train_rounds = 650;

	private readonly eta = 0.6;

	constructor(seed: number) {
		this.rng = new ml.PCG32(BigInt(128 * seed));
		this.network = new Network(seed, this.input_size, [80, 40, 10]);
		this.data = new MnistData();
		this.data.load();
	}

	getImage(id: number) {
		const img_start = id * this.input_size;
		const img_end = img_start + this.input_size;

		const label_start = id * this.output_size;
		const label_end = label_start + this.output_size;

		const img: Batch = {
			xs: new ml.Vector(this.data.trainImages.slice(img_start, img_end)),
			ys: new ml.Vector(new Float32Array(this.data.trainLabels.slice(label_start, label_end)))
		};
		return img;
	}

	evaluate(input: ml.Vector) {
		return this.network.evaluate(input);
	}

	transformImage(image: ml.Vector): ml.Vector {
		const augmented = new Float32Array(image.elements.length);
		const width = this.img_width;

		// Random translation offsets (-4 to 4 pixels)
		const dx = Math.floor(Math.random() * 5) - 2;
		const dy = Math.floor(Math.random() * 5) - 2;

		// Random rotation angle (-30 to 30 degrees)
		const angle = (Math.random() * 30 - 15) * (Math.PI / 180);

		// Noise level (0 to 0.1)
		const noiseLevel = Math.random() * 0.1;

		// Transformation calculations
		const cosA = Math.cos(angle);
		const sinA = Math.sin(angle);
		const cx = width / 2;
		const cy = width / 2;

		for (let y = 0; y < width; y++) {
			for (let x = 0; x < width; x++) {
				// Translate and rotate
				const nx = Math.floor(
					(cosA * (x - cx) - sinA * (y - cy) + cx + dx)
				);
				const ny = Math.floor(
					(sinA * (x - cx) + cosA * (y - cy) + cy + dy)
				);

				// Check bounds and assign pixel values
				if (nx >= 0 && nx < width && ny >= 0 && ny < width) {
					augmented[y * width + x] = image.elements[ny * width + nx];
				}

				// Add random noise
				augmented[y * width + x] += noiseLevel * (Math.random() - 0.5);
				augmented[y * width + x] = Math.min(1, Math.max(0, augmented[y * width + x])); // Clip values to [0, 1]
			}
		}

		return new ml.Vector(augmented);
	}

	train() {
		console.time("train");
		for (let e = 0; e < this.num_epochs; e++) {
			console.log("========================================")
			console.log("EPOCH: " + e);

			var avg_cost = 0.0;
			var avg_perc = 0.0;
			for (let r = 0; r < this.train_rounds; r++) {
				const batches = [];
				for (let i = 0; i < this.batch_size; i++) {
					const batch = this.getImage(r * this.batch_size + i);
					const transformed = this.transformImage(batch.xs);
					batches.push({ xs: transformed, ys: batch.ys });
				}

				const ret = this.network.processMiniBatch(batches, this.eta);
				avg_cost += ret.avg_cost;
				avg_perc += ret.num_correct / this.batch_size;
				if (r % 100 == 0)
					console.log("ON ROUND = ", r);
			}
			avg_cost /= this.train_rounds;
			avg_perc /= this.train_rounds;
			console.log("avg round cost =", Number(avg_cost.toFixed(4)), "avg training % =", avg_perc);
		}
		console.timeEnd("train");
	}

	renderImage(img: ml.Vector, ctx: CanvasRenderingContext2D) {
		ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

		ctx.canvas.width = 800;
		ctx.canvas.height = 800;

		const width = ctx.canvas.width / this.img_width;
		const height = ctx.canvas.height / this.img_width;

		ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

		var iter = 0;
		for (let row = 0; row < this.img_width; row++) {
			for (let col = 0; col < this.img_width; col++) {
				iter++;
				const pixelIndex = row * this.img_width + col;
				const intensity = img.elements[pixelIndex];

				const grayscale = Math.floor(intensity * 255);
				ctx.fillStyle = `rgb(${grayscale}, ${grayscale}, ${grayscale})`;
				ctx.fillRect(col * width, row * height, width, height);
			}
		}

	}

}
