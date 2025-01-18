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

	private readonly batch_size = 100;
	private readonly train_rounds = 100;

	private readonly eta = 1.0;

	constructor(seed: number) {
		this.network = new Network(seed, this.input_size, 3, [16, 16, 10]);
		this.data = new MnistData();
	}

	evaluate(input: ml.Vector) {
		return this.network.feedforward(input);
	}

	train() {
		for (let r = 0; r < this.train_rounds; r++) {
			const batches = [];
			for (let i = 0; i < this.batch_size; i++) {
				batches.push({
					xs: new ml.Vector(this.data.trainImages.slice(this.input_size * i, this.input_size)),
					ys: new ml.Vector(new Float32Array(this.data.trainLabels.slice(this.output_size * i, this.output_size)))
				});
			}

			this.network.processMiniBatch(batches, this.eta);

		}


	}

}
