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

	//processMiniBatch(mini_batch: something, eta: number) {
	//
	//}

	feedforward(input: ml.Vector): ml.Vector {
		var output = input.clone();
		for (let i = 0; i < this.layers.length; i++) {
			output = this.layers[i].feedforward(output);
		}
		return output;
	}

	processPartials(input: ml.Vector, target: ml.Vector) {
		var activation = this.feedforward(input);

		const sp = Layer.sigmoid_prime(activation);
		var delta = ml.Vector.hadamard(this.cost_derivative(activation, target), sp);

		const nabla_b = [];
		const nabla_w = [];

		for (let i = this.layers.length - 1; i >= 0; i--) {
			const prev_activations = i == 0 ? input : Layer.sigmoid(this.layers[i - 1].zs);
			const ret = this.layers[i].backprop(prev_activations, delta);
			delta = ret.delta;
			nabla_b.unshift(ret.nabla_b);
			nabla_w.unshift(ret.nabla_w);
		}

		return { nabla_b, nabla_w };

	}

	// derivative of 1/2 * (a - y)^2
	cost_derivative(activations: ml.Vector, target: ml.Vector) {
		return ml.Vector.sub(activations, target);
	}


}

export class Layer {
	weights: ml.Matrix; // to prev neurons
	biases: ml.Vector; // when being activated from prev neurons

	//nabla_w: ml.Vector;
	//nabla_b: ml.Vector;

	zs: ml.Vector; // raw activations (without sigmoid)

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

		//this.nabla_w = new ml.Vector(new Float32Array(num_neurons));
		//this.nabla_b = new ml.Vector(new Float32Array(num_neurons));

		this.zs = ml.Vector.fromType(this.biases.type, this.biases.length);
	}

	// 1 / (1 + e^-x)
	static sigmoid(activations: ml.Vector): ml.Vector {
		const sigmoid_x = (num: number) => {
			return 1 / (1 + Math.exp(-num));
		}

		const output = activations.clone();
		for (let i = 0; i < output.length; i++) {
			output.elements[i] = sigmoid_x(output.elements[i]);
		}
		return output;
	}

	static sigmoid_prime(activations: ml.Vector): ml.Vector {
		const sigmoid_prime_x = (num: number) => {
			const sig = (v: number) => {
				return 1 / (1 + Math.exp(-v));
			}
			const s = sig(num);
			return s * (1 - s);
		}

		const output = activations.clone();
		for (let i = 0; i < output.length; i++) {
			output.elements[i] = sigmoid_prime_x(output.elements[i]);
		}
		return output;
	}

	/**
	* @returns {ml.Vector} the next activations
	*/
	feedforward(prev_activations: ml.Vector): ml.Vector {
		const raw = ml.Matrix.affineTransformation(this.weights, prev_activations, this.biases);
		this.zs = raw;
		return Layer.sigmoid(raw);
	}

	backprop(prev_activations: ml.Vector, delta: ml.Vector): {
		delta: ml.Vector, nabla_b: ml.Vector, nabla_w: ml.Vector
	} {
		const sp = Layer.sigmoid_prime(this.zs);
		const new_delta = ml.Vector.hadamard(ml.Matrix.multTransposeVector(this.weights, delta), sp);
		const nabla_b = new_delta;
		const nabla_w = ml.Vector.hadamard(prev_activations, new_delta);
		return { delta: new_delta, nabla_b, nabla_w };
	}


}
