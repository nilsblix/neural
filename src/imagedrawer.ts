import * as ml from "./mathlib/math.ts";

export class ImageDrawer {
	image_input: ml.Vector;
	mode: "draw" | "erase";

	constructor(type: "f32" | "f64", image_vec_length: number) {
		this.mode = "draw";
		this.image_input = ml.Vector.fromType(type, image_vec_length);
	}

	updateInput(canv_position: { x: number, y: number }, cell_width: number, brush_size: number) {
		const id_x = Math.floor(canv_position.x / cell_width);
		const id_y = Math.floor(canv_position.y / cell_width);

		const side_length = Math.sqrt(this.image_input.length);
		if (id_x < 0 || id_x >= side_length)
			return;
		if (id_y < 0 || id_y >= side_length)
			return;

		for (let col = Math.floor(id_x - brush_size); col <= id_x + brush_size; col++) {
			for (let row = Math.floor(id_y - brush_size); row <= id_y + brush_size; row++) {
				const idx = row * cell_width + col;

				const dist = Math.sqrt((col - id_x) ** 2 + (row - id_y) ** 2);

				if (dist <= brush_size) {
					switch (this.mode) {
						case "draw":
							const draw_value = Math.max(0, 1 - dist / brush_size);
							this.image_input.elements[idx] = Math.min(1.0, this.image_input.elements[idx] + draw_value * 0.06);
							break;
						case "erase":
							this.image_input.elements[idx] = 0;
							break;
					}

				}
			}

		}
	}
}
