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

	private readonly input_size = 28 * 28;
	private readonly output_size = 10;

	private readonly batch_size = 200;
	private readonly train_rounds = 40;

	private readonly eta = 0.4;

	constructor(seed: number) {
		this.network = new Network(seed, this.input_size, [40, 16, 10]);
		this.data = new MnistData();
		this.data.load();
	}

	evaluate(input: ml.Vector) {
		return this.network.feedforward(input);
	}

	train() {
		for (let r = 0; r < this.train_rounds; r++) {
			const batches = [];
			for (let i = 0; i < this.batch_size; i++) {
				const img_start = (r + i) * this.input_size;
				const img_end = img_start + this.input_size;

				const label_start = (r + i) * this.output_size;
				const label_end = label_start + this.output_size;
				batches.push({
					xs: new ml.Vector(this.data.trainImages.slice(img_start, img_end)),
					ys: new ml.Vector(new Float32Array(this.data.trainLabels.slice(label_start, label_end)))
				});
				console.log("ANSWERS: ", batches[batches.length - 1].ys);
			}

			console.log(`start of batch=${r}`);
			this.network.processMiniBatch(batches, this.eta);

		}


	}

	renderImage(image_id: number, ctx: CanvasRenderingContext2D) {
		const img_start = image_id * this.input_size;
		const img_end = img_start + this.input_size;

		const label_start = image_id * this.output_size;
		const label_end = label_start + this.output_size;

		const img: Batch = {
			xs: new ml.Vector(this.data.trainImages.slice(img_start, img_end)),
			ys: new ml.Vector(new Float32Array(this.data.trainLabels.slice(label_start, label_end)))
		};

		// continue here! use fillrects to draw the square (28 x 28).

		//// Set the canvas size to 28x28 for the MNIST image
		const width = 28;
		const height = 28;
		ctx.canvas.width = width;
		ctx.canvas.height = height;

		// Scale the image to fit larger canvas if needed
		const scale = 10; // Change this to adjust the scaling factor
		ctx.canvas.style.width = `${width * scale}px`;
		ctx.canvas.style.height = `${height * scale}px`;

		// Clear the canvas
		ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

		// Draw the image using fillRect for each pixel
		for (let row = 0; row < height; row++) {
			for (let col = 0; col < width; col++) {
				// Get the pixel intensity (normalized to [0, 1])
				const pixelIndex = row * width + col;
				const intensity = img.xs.elements[pixelIndex];

				// Convert intensity to a grayscale color
				const grayscale = Math.floor(intensity * 255);

				// Set the fill color
				ctx.fillStyle = `rgb(${grayscale}, ${grayscale}, ${grayscale})`;

				// Draw the pixel as a small square
				ctx.fillRect(col, row, 1, 1);
			}
		}

	}

}
