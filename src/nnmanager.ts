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

}
