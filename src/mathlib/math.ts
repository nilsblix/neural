export class PCG32 {
	private state: bigint;
	private readonly multiplier: bigint = 6364136223846793005n;
	private readonly increment: bigint = 1442695040888963407n;

	constructor(seed: bigint) {
		this.state = seed + this.increment;
		this.pcg32(); // Advance the state once to avoid starting from zero
	}

	private rotr32(x: number, r: number): number {
		return (x >>> r) | (x << (-r & 31)) >>> 0;
	}

	public pcg32(): number {
		const x = this.state;
		const count = Number((x >> 59n) & 31n); // Extract the top 5 bits as the rotation amount

		this.state = x * this.multiplier + this.increment;

		let xorShifted = Number((x >> 18n) & 0xFFFFFFFFn); // 18 = (64 - 32 - 5)
		xorShifted ^= xorShifted >>> 18; // Xorshift for randomness

		this.step();

		return this.rotr32(xorShifted, count);
	}

	public step(): void {
		this.state++;
	}

	public pcg32_0to1(): number {
		return (1 + this.pcg32() / 4294967296) / 2;
	}
}

export class V2 {
	x: number;
	y: number;
	constructor(x = 0.0, y = 0.0) {
		this.x = x;
		this.y = y;
	}

	static add(a: V2, b: V2) {
		return new V2(a.x + b.x, a.y + b.y);
	}

	static sub(a: V2, b: V2) {
		return new V2(a.x - b.x, a.y - b.y);
	}

	static mult(s: number, v: V2) {
		return new V2(s * v.x, s * v.y);
	}

	static dot(a: V2, b: V2) {
		return a.x * b.x + a.y * b.y;
	}

	len_squared() {
		return V2.dot(this, this);
	}

	len() {
		return Math.sqrt(this.len_squared());
	}
}

export class Vector {
	elements: Float32Array | Float64Array;
	type: "f32" | "f64";

	constructor(elements: Float32Array | Float64Array) {
		this.elements = elements;
		this.type = elements instanceof Float32Array ? "f32" : "f64";
	}

	get length() {
		return this.elements.length;
	}

	clone() {
		const ret = Vector.fromType(this.type, this.length);
		for (let i = 0; i < this.length; i++) {
			ret.elements[i] = this.elements[i];
		}
		return ret;
	}

	static fromType(type: "f32" | "f64", length: number): Vector {
		const ret = type == "f32" ? new Float32Array(length) : new Float64Array(length);
		return new Vector(ret);
	}

	static checkCompatibility(a: Vector, b: Vector) {
		if (a.length != b.length) {
			throw new Error(
				`Vector lengths do not match: a.len=${a.length}, b.len=${b.length}`
			);
		}

		if (a.type != b.type) {
			throw new Error(
				`Vector types do not match: a.type=${a.type}, b.type=${b.type}`
			);
		}
	}

	static add(a: Vector, b: Vector) {
		Vector.checkCompatibility(a, b);

		const ret = Vector.fromType(a.type, a.length);
		for (let i = 0; i < a.length; i++) {
			ret.elements[i] = a.elements[i] + b.elements[i];
		}

		return ret;
	}


	static sub(a: Vector, b: Vector) {
		Vector.checkCompatibility(a, b);

		const ret = Vector.fromType(a.type, a.length);
		for (let i = 0; i < a.length; i++) {
			ret.elements[i] = a.elements[i] - b.elements[i];
		}

		return ret;
	}


	static dot(a: Vector, b: Vector) {
		if (a.length != b.length) {
			throw new Error(
				`Vector lengths do not match: a.len=${a.length}, b.len=${b.length}`
			);
		}

		if (a.type != b.type) {
			throw new Error(
				`Vector types do not match: a.type=${a.type}, b.type=${b.type}`
			);
		}

		let ret = 0.0;
		for (let i = 0; i < a.length; i++) {
			ret += a.elements[i] * b.elements[i];
		}

		return ret;
	}


	static scale(s: number, vec: Vector) {
		const ret = Vector.fromType(vec.type, vec.length);
		for (let i = 0; i < ret.length; i++) {
			ret.elements[i] *= s;
		}
		return ret;
	}
}

/**
 * Default type is "f32".
 * Matrix is structured as: (2x2) => TL: 0,0 | BL: 1,0 | TR: 0,1 | BR: 1,1
 */
export class Matrix {
	elements: Vector[];
	type: "f32" | "f64";

	constructor(elements: Vector[]) {
		if (elements.length > 0) {
			this.elements = elements;
			this.type = elements[0].type;
		} else {
			this.type = "f32";
			this.elements = [];
		}
	}

	/**
	* Get the vertical length
	*/
	get i_length() {
		return this.elements.length;
	}

	/**
	* Get the horizontal length
	*/
	j_length(i: number) {
		if (i < 0 || i > this.elements.length - 1) {
			throw new Error(
				`Tried to j_index outside of i_bounds. i_index=${i}, i_length=${this.i_length}`
			)
		}
		return this.elements[i].length;
	}

	static matrix_vector_compatibility(mat: Matrix, vec: Vector) {
		if (mat.j_length(0) != vec.length) {
			throw new Error(`Matrix and Vector length are not compatible for multiplying. Matrix.j_length=${mat.j_length(0)}. Vector.length=${vec.length}`)
		}

		if (mat.type != vec.type) {
			throw new Error(`Matrix and Vector types are not compatible for multiplying. Matrix.type=${mat.type}. Vector.type=${vec.type}`)
		}
	}

	static matrix_transpose_vector_compatibility(mat: Matrix, vec: Vector) {
		if (mat.i_length != vec.length) {
			throw new Error(`Matrix and Vector length are not compatible for multiplying. Matrix.i_length=${mat.i_length}. Vector.length=${vec.length}`)
		}

		if (mat.type != vec.type) {
			throw new Error(`Matrix and Vector types are not compatible for multiplying. Matrix.type=${mat.type}. Vector.type=${vec.type}`)
		}
	}

	static multVector(mat: Matrix, vec: Vector) {
		Matrix.matrix_vector_compatibility(mat, vec);

		const ret = Vector.fromType(vec.type, mat.i_length);

		for (let i = 0; i < mat.i_length; i++) {
			ret.elements[i] = Vector.dot(mat.elements[i], vec);
		}
		return ret;
	}

	static affineTransformation(mat: Matrix, vec: Vector, off: Vector): Vector {
		return Vector.add(Matrix.multVector(mat, vec), off);
	}

	static multTransposeVector(mat: Matrix, vec: Vector) {
		Matrix.matrix_transpose_vector_compatibility(mat, vec);

		const ret = Vector.fromType(vec.type, mat.j_length(0));

		for (let j = 0; j < mat.j_length(0); j++) {
			let dot = 0.0;
			for (let i = 0; i < mat.i_length; i++) {
				dot += mat.elements[i].elements[j] * vec.elements[i];
			}
			ret.elements[j] = dot;
		}
		return ret;
	}
}
