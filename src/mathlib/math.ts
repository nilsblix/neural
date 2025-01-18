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
