import * as ml from "./mathlib/math.ts";

export class ImageDrawer {
	image_input: ml.Vector;
	mode: "draw" | "erase";

	constructor(type: "f32" | "f64", image_vec_length: number) {
		this.mode = "draw";
		this.image_input = ml.Vector.fromType(type, image_vec_length);
	}

	updateInput(canv_position: { x: number, y: number }, cell_width: number) {
		const id_x = Math.floor(canv_position.x / cell_width);
		const id_y = Math.floor(canv_position.y / cell_width);

		switch (this.mode) {
			case "draw":
				for (let row = id_x - 1; row <= id_x + 1; row++) {
					for (let col = id_y - 1; col <= id_y + 1; col++) {
						const middle = row == id_x && col == id_y;
						var value = this.image_input.elements[row * cell_width + col];
						this.image_input.elements[row * cell_width + col] = middle ? 1.0 : Math.min(1.0, value + 0.03);
					}
				}
				break;
			case "erase":

				for (let row = id_x - 1; row <= id_x + 1; row++) {
					for (let col = id_y - 1; col <= id_y + 1; col++) {
						//const middle = row == id_x && col == id_y;
						this.image_input.elements[row * cell_width + col] = 0.0;
					}
				}
				break;
		}
	}
}
