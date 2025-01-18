import * as ml from "./mathlib/math.ts";
import { Network } from "./neural_network/nnetwork";
import { MnistData } from "./mnist";

export class Engine {
	network: Network;
	data: MnistData;

	constructor(seed: number) {
		this.network = new Network(seed, 28 * 28, 3, [16, 16, 10]);
		this.data = new MnistData();
	}

	evaluate(input: ml.Vector) {
		return this.network.feedforward(input);
	}

	train() {
		const batch_size = 100;
	}

}
