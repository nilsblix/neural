import * as ml from "./mathlib/math.ts";

export class ImageDrawer {
	image_input: ml.Vector;
	mode: "draw" | "erase";

	constructor(type: "f32" | "f64", image_vec_length: number) {
		this.mode = "draw";
		this.image_input = ml.Vector.fromType(type, image_vec_length);
	}

	updateInput(canv_position: { x: number, y: number }, num_cells: number, brush_size: number, add_rate: number) {
		const id_x = Math.floor(canv_position.x / num_cells);
		const id_y = Math.floor(canv_position.y / num_cells);

		if (id_x < 0 || id_x >= num_cells)
			return;
		if (id_y < 0 || id_y >= num_cells)
			return;

		for (let col = Math.floor(id_x - brush_size); col <= id_x + brush_size; col++) {
			for (let row = Math.floor(id_y - brush_size); row <= id_y + brush_size; row++) {
				const idx = row * num_cells + col;

				const dist = Math.sqrt((col - id_x) ** 2 + (row - id_y) ** 2);

				if (dist <= brush_size) {
					switch (this.mode) {
						case "draw":
							const draw_value = Math.max(0, 1 - dist / brush_size);
							this.image_input.elements[idx] = Math.min(1.0, this.image_input.elements[idx] + draw_value * add_rate);
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
