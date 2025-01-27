import * as ml from "../mathlib/math.ts";
import { Batch } from "../nnmanager.ts";

export class Network {
	layers: Layer[];

	static readonly object_type: {
		layers: Array<typeof Layer.object_type>;
	}

	/**
	 * layers goes from the one after input to (includes) output
	 * layers do not contain any information about activation, the activations just gets passed through the network
	 */
	constructor(seed: number, input_size: number, layer_sizes: number[]) {
		this.layers = [];

		const rng = new ml.PCG32(BigInt(seed));
		for (let l = 0; l < layer_sizes.length; l++) {
			this.layers.push(new Layer(rng, layer_sizes[l], l == 0 ? input_size : layer_sizes[l - 1]));
		}
	}

	toObject(): typeof Network.object_type {
		const arr = [];
		for (let i = 0; i < this.layers.length; i++) {
			arr.push(this.layers[i].toObject());
		}
		return {
			layers: arr
		}
	}

	static fromObject(data: typeof Network.object_type) {
		const net = Network.empty();
		const lays = [];
		for (let i = 0; i < data.layers.length; i++) {
			lays.push(Layer.fromObject(data.layers[i]));
		}
		net.layers = lays;
		return net;
	}

	static empty() {
		return new Network(0, 0, []);
	}

	evaluateRaw(input: ml.Vector) {
		const output = this.feedforward(input);
		return output;
	}

	evaluateSoftmaxxed(input: ml.Vector) {
		const output = this.evaluateRaw(input);
		return this.softmax(output);
	}

	processMiniBatch(mini_batch: Batch[], eta: number) {
		const x1 = mini_batch[0].xs;
		const y1 = mini_batch[0].ys;

		const ret = this.processPartials(x1, y1);
		const nb = ret.nabla_b;
		const nw = ret.nabla_w;

		var avg_cost = 0.0;
		var num_correct = 0;
		var num_wrong = 0;

		for (let i = 0; i < mini_batch.length; i++) {
			const x = mini_batch[i].xs;
			const y = mini_batch[i].ys;
			const { correct, cost, nabla_b, nabla_w } = this.processPartials(x, y);

			avg_cost += cost;

			if (correct)
				num_correct++;
			else
				num_wrong++;

			for (let k = 0; k < this.layers.length; k++) {
				nb[k] = ml.Vector.add(nb[k], nabla_b[k]);
				nw[k] = ml.Matrix.add(nw[k], nabla_w[k]);
			}
		}

		avg_cost /= mini_batch.length;

		console.assert(this.layers.length == nw.length);
		console.assert(this.layers.length == nb.length);
		const cnst = eta / mini_batch.length;

		for (let i = 0; i < this.layers.length; i++) {
			this.layers[i].weights = ml.Matrix.sub(this.layers[i].weights, ml.Matrix.scale(cnst, nw[i]));
			this.layers[i].biases = ml.Vector.sub(this.layers[i].biases, ml.Vector.scale(cnst, nb[i]));
		}

		return { avg_cost, num_correct };
	}

	feedforward(input: ml.Vector): ml.Vector {
		var output = input.clone();
		for (let i = 0; i < this.layers.length; i++) {
			output = this.layers[i].feedforward(output);
		}
		return output;
	}

	processPartials(input: ml.Vector, target: ml.Vector) {
		var activation = this.feedforward(input);
		const cost = this.cost(activation, target);
		const correct = this.correct(activation, target);

		const sp = Layer.sigmoid_prime(activation);
		var delta = ml.Vector.hadamard(this.cost_derivative(activation, target), sp);

		const nabla_b = [delta];
		const nabla_w = [ml.Matrix.fromVecMultVec(delta, this.layers[this.layers.length - 2].zs)];

		for (let i = this.layers.length - 2; i >= 0; i--) {
			const prev_activations = i == 0 ? input : Layer.sigmoid(this.layers[i - 1].zs);
			const ret = this.layers[i].backprop(prev_activations, delta, this.layers[i + 1].weights);
			delta = ret.delta;
			nabla_b.unshift(ret.nabla_b);
			nabla_w.unshift(ret.nabla_w);
		}

		return { correct, cost, nabla_b, nabla_w };

	}

	correct(activations: ml.Vector, target: ml.Vector) {
		var target_correct = -1;

		var activations_guess_correct = -1;
		var highest_activation = Number.NEGATIVE_INFINITY;
		for (let i = 0; i < activations.length; i++) {
			if (activations.elements[i] > highest_activation) {
				highest_activation = activations.elements[i];
				activations_guess_correct = i;
			}

			if (target.elements[i] == 1)
				target_correct = i;
		}
		return target_correct == activations_guess_correct;
	}



	cost(activations: ml.Vector, target: ml.Vector) {
		const dv = ml.Vector.sub(activations, target);
		return 1 / 2 * ml.Vector.dot(dv, dv);
	}

	// derivative of 1/2 * (a - y)^2
	cost_derivative(activations: ml.Vector, target: ml.Vector) {
		return ml.Vector.sub(activations, target);
	}

	softmax(activations: ml.Vector) {
		const TEMP = 100;
		var sum = 0.0;
		for (let i = 0; i < activations.length; i++) {
			const act = activations.elements[i];
			sum += Math.exp(act * TEMP);
		}
		const vec = ml.Vector.fromType(activations.type, activations.length);
		for (let i = 0; i < activations.length; i++) {
			vec.elements[i] = Math.exp(activations.elements[i] * TEMP) / sum;
		}
		return vec;
	}

}

export class Layer {
	weights: ml.Matrix; // to prev neurons
	biases: ml.Vector; // when being activated from prev neurons

	zs: ml.Vector; // raw activations (without sigmoid)

	static readonly object_type: {
		weights: typeof ml.Matrix.object_type;
		biases: typeof ml.Vector.object_type;
		zs: typeof ml.Vector.object_type;
	}

	// weights and biases gets initialized to [-1, 1] ?
	constructor(rng: null | ml.PCG32, num_neurons: number, prev_num_neurons: number) {
		if (rng == null) {
			this.weights = new ml.Matrix([])
			const empty_vec = new ml.Vector(new Float32Array([]));
			this.biases = empty_vec.clone();
			this.zs = empty_vec.clone();
			return;
		}
		const elem = [];
		const bs = [];
		for (let i = 0; i < num_neurons; i++) {
			const w_i = [];
			for (let k = 0; k < prev_num_neurons; k++) {
				w_i.push(Math.random() * 2 - 1);
			}
			const weight_vec = new ml.Vector(new Float32Array(w_i));
			elem.push(weight_vec);

			bs.push(Math.random() * 2 - 1);
		}
		this.weights = new ml.Matrix(elem);
		this.biases = new ml.Vector(new Float32Array(bs));

		this.zs = ml.Vector.fromType(this.biases.type, this.biases.length);
	}

	toObject(): typeof Layer.object_type {
		return {
			weights: this.weights.toObject(),
			biases: this.biases.toObject(),
			zs: this.zs.toObject(),
		}
	}

	static fromObject(data: typeof Layer.object_type) {
		const lay = Layer.empty();
		lay.weights = ml.Matrix.fromObject(data.weights);
		lay.biases = ml.Vector.fromObject(data.biases);
		lay.zs = ml.Vector.fromObject(data.zs);
		return lay;
	}

	static empty() {
		return new Layer(null, 0, 0);
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

	backprop(prev_activations: ml.Vector, delta: ml.Vector, next_weights: ml.Matrix): {
		delta: ml.Vector, nabla_b: ml.Vector, nabla_w: ml.Matrix
	} {
		const sp = Layer.sigmoid_prime(this.zs);
		const new_delta = ml.Vector.hadamard(ml.Matrix.multTransposeVector(next_weights, delta), sp);
		const nabla_b = new_delta.clone();
		const nabla_w = ml.Matrix.fromVecMultVec(new_delta, prev_activations);
		return { delta: new_delta, nabla_b, nabla_w };
	}

}
