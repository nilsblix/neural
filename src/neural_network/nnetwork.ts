import * as ml from "../mathlib/math.ts";

export class Network {
	layers: Layer[];

	/**
	 * layers goes from the one after input to (includes) output
	 * layers do not contain any information about activation, the activations just gets passed through the network
	 */
	constructor(seed: number, input_size: number, num_layers: number, layer_sizes: number[]) {
		if (num_layers != layer_sizes.length) {
			throw new Error(`num_layers and number of sizes don't match in network constructor. num_layers=${num_layers}, num_sizes=${layer_sizes.length}`);
		}

		this.layers = [];

		const rng = new ml.PCG32(BigInt(seed));
		for (let l = 0; l < num_layers; l++) {
			this.layers.push(new Layer(rng, layer_sizes[l], l == 0 ? input_size : layer_sizes[l - 1]));
		}
	}

	outputFromInput(input: ml.Vector): ml.Vector {
		var output = input.clone();
		for (let i = 0; i < this.layers.length; i++) {
			output = this.layers[i].feedForward(output);
		}
		return output;
	}


}

export class Layer {
	weights: ml.Matrix; // to prev neurons
	biases: ml.Vector; // when being activated from prev neurons

	// weights and biases gets initialized to [-1, 1] ?
	constructor(rng: ml.PCG32, num_neurons: number, prev_num_neurons: number) {
		const elem = [];
		const bs = [];
		for (let i = 0; i < num_neurons; i++) {
			const w_i = [];
			for (let k = 0; k < prev_num_neurons; k++) {
				w_i.push(rng.pcg32_0to1() * 2 - 1);
			}
			const weight_vec = new ml.Vector(new Float32Array(w_i));
			elem.push(weight_vec);

			bs.push(rng.pcg32_0to1() * 2 - 1);
		}
		this.weights = new ml.Matrix(elem);
		this.biases = new ml.Vector(new Float32Array(bs));
	}

	sigma(activations: ml.Vector): ml.Vector {
		const output = activations.clone();
		for (let i = 0; i < output.length; i++) {
			// 1 / (1 + e^-x)
			output.elements[i] = 1 / (1 + Math.exp(-output.elements[i]));
		}
		return output;
	}

	/**
	* @returns {ml.Vector} the next activations
	*/
	feedForward(prev_activations: ml.Vector): ml.Vector {
		const raw = ml.Matrix.affineTransformation(this.weights, prev_activations, this.biases);
		return this.sigma(raw);
	}


}
